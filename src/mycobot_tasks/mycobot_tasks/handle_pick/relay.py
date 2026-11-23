"""Relay through a spot on the desk: shelf D11 <-> TurtleBot (n / j).

D11 is reachable only with the reversed arm (J1 ~180 deg from the handle: the upper arm of
the normal pose hits the shelf rack), and with the tray held there is no way from D11 to the
TurtleBot (study notes/reach_study_2026-11-20). So the tray goes over the desk:

  D11 (reversed arm) -> lift, pull out, up -> tray turned in place -> carried with the tool
  tilted <= --relay-tilt over a 2 cm grid (Dijkstra) -> straightened, put down on a desk
  spot -> empty arm switched to the normal pose (OMPL, the tray on the desk an obstacle) ->
  regrasped (fingers turned 180 deg when that is what works: J6 stays within +-180) ->
  turned, carried tilted over the TurtleBot box -> lowered tilted, straightened, put in.

TurtleBot -> D11 is the same motion backwards. Everything is planned before anything moves,
from the measured poses (TurtleBot marker / tray handle markers): fast numeric IK on the
URDF (scipy) + MoveIt collision checks with the held tray attached to the TCP, then the
joint paths run through move_group's execute_trajectory.
"""
import heapq
import math
import os
import shlex
import time

import numpy as np
from scipy.optimize import least_squares

from mycobot_calibration.geometry import rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import _wait
from mycobot_tasks.red_cube_hover import DOWN, Stop

# joint windows [rad] = mycobot_bridge driver.yaml joint_min/max_deg (J6 +-180)
LO = np.radians([-168.0, -135.0, -145.0, -145.0, -168.0, -180.0])
HI = -LO
TOUCH = ['gripper_connection', 'gripper_base', 'gripper_left1', 'gripper_left2', 'gripper_left3',
         'gripper_right1', 'gripper_right2', 'gripper_right3', 'gripper_tcp']
TRAY_ID = 'relay_tray'
# what the tray stands on (place_scene names): its contacts with these are normal there
D11_ON, DESK_ON, BOX_ON = ('shelf_fork',), ('desk',), ('bot_box',)
TMP_ID = 'relay_tray_plan'
# empty arm travelling into / out of D11: fingers closed (inside the held tray's envelope,
# the way it was checked with the tray); opened over the handle only
CLOSED = 0.0


def seg(a, b, step=0.005):
    """Points from a (excluded) to b (included) every <= step."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step)))
    return [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]


class RelayFail(Exception):
    pass


class RelayMixin:
    # ------------------------------------------------------------------ setup
    def relay_setup(self):
        if getattr(self, '_rl_arm', None) is not None:
            return
        from moveit_msgs.srv import GetMotionPlan, GetStateValidity
        from mycobot_calibration.kin_calib import Arm
        from mycobot_tasks import place_scene as ps
        self._rl_arm = Arm(self._relay_urdf(), tip='gripper_tcp')
        self._rl_sv = self.node.create_client(GetStateValidity, 'check_state_validity')
        self._rl_plan = self.node.create_client(GetMotionPlan, 'plan_kinematic_path')
        if not (self._rl_sv.wait_for_service(timeout_sec=5.0) and
                self._rl_plan.wait_for_service(timeout_sec=5.0)):
            raise Stop('move_group check_state_validity / plan_kinematic_path not available')
        self._rl_rng = np.random.default_rng(0)
        self._rl_world = set(ps.NAMES) | {TRAY_ID, TMP_ID, 'bot_keepout', 'calibration_table'}
        self._rl_attached = False

    def _relay_urdf(self):
        """robot_description of the running robot_state_publisher (same calibration as
        move_group), else --relay-urdf, else the xacro without a calibration file."""
        from rcl_interfaces.srv import GetParameters
        cli = self.node.create_client(GetParameters, '/robot_state_publisher/get_parameters')
        if cli.wait_for_service(timeout_sec=3.0):
            res = _wait(cli.call_async(GetParameters.Request(names=['robot_description'])), 5.0)
            if res is not None and res.values and res.values[0].string_value:
                return res.values[0].string_value
        if self.args.relay_urdf:
            with open(os.path.expanduser(self.args.relay_urdf)) as f:
                return f.read()
        import subprocess
        from ament_index_python.packages import get_package_share_directory
        x = os.path.join(get_package_share_directory('mycobot_description'), 'urdf',
                         'mycobot_320_pi_2022', 'pro_320_pi_2022_force_gripper.urdf.xacro')
        print('relay: no robot_description from robot_state_publisher: xacro without a '
              'calibration file', flush=True)
        return subprocess.run(['xacro', x], capture_output=True, text=True, check=True).stdout

    # ------------------------------------------------------------------ geometry
    def _tip_to_bottom(self):
        a = self.args
        return a.tray_height - a.grasp_depth          # finger tips over the tray bottom

    def _tray_attached(self, pad=0.0):
        """The held tray as an AttachedCollisionObject on gripper_tcp (TCP z down), grown by
        pad on every side (clearance check)."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from shape_msgs.msg import SolidPrimitive
        L, W, H = self.args.relay_tray
        ac = AttachedCollisionObject(link_name='gripper_tcp', touch_links=TOUCH)
        o = ac.object
        o.id, o.header.frame_id, o.operation = TRAY_ID, 'gripper_tcp', CollisionObject.ADD
        o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                       dimensions=[L + 2 * pad, W + 2 * pad, H + 2 * pad])]
        p = Pose()
        p.position.z = self._tip_to_bottom() - H / 2      # body from the bottom up H
        p.orientation.w = 1.0
        o.primitive_poses = [p]
        return ac

    def _tray_world(self, q, oid=TRAY_ID, pad=0.0):
        """World collision object of the tray where the TCP at joints q holds it (grown by
        pad on every side)."""
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import CollisionObject
        from scipy.spatial.transform import Rotation
        from shape_msgs.msg import SolidPrimitive
        L, W, H = self.args.relay_tray
        T = self._rl_arm.fk(q)
        c = T[:3, 3] + T[:3, 2] * (self._tip_to_bottom() - H / 2) + [0, 0, 0.002]
        o = CollisionObject()
        o.id, o.header.frame_id, o.operation = oid, 'base', CollisionObject.ADD
        o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                       dimensions=[L + 2 * pad, W + 2 * pad, H + 2 * pad])]
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in c)
        qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = \
            float(qx), float(qy), float(qz), float(qw)
        o.primitive_poses = [p]
        return o

    def _scene(self, world=(), attached=()):
        from moveit_msgs.msg import PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
        req.scene.robot_state.is_diff = True
        req.scene.world.collision_objects = list(world)
        req.scene.robot_state.attached_collision_objects = list(attached)
        res = _wait(self.mv.scene.call_async(req), 5.0)
        return res is not None and res.success

    def _remove_world(self, oid):
        from moveit_msgs.msg import CollisionObject
        o = CollisionObject()
        o.id, o.header.frame_id, o.operation = oid, 'base', CollisionObject.REMOVE
        self._scene(world=[o])

    def relay_clear(self):
        """No relay tray left in the scene (world copy; the attached one only if ours)."""
        for oid in (TRAY_ID, TMP_ID):
            self._remove_world(oid)
        if self._rl_attached:
            self._detach(None)

    def _attach(self):
        self._remove_world(TRAY_ID)
        self._scene(attached=[self._tray_attached()])
        self._rl_attached = True

    def _detach(self, q):
        """Held tray -> world object where it stands (q None: just drop it)."""
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        ac = AttachedCollisionObject(link_name='gripper_tcp')
        ac.object.id, ac.object.operation = TRAY_ID, CollisionObject.REMOVE
        self._scene(attached=[ac])
        self._rl_attached = False
        self._remove_world(TRAY_ID)          # MoveIt may have put it back into the world
        if q is not None:
            self._scene(world=[self._tray_world(q)])

    def _desk(self):
        """(desk box centre xy, size xy, yaw, desk top z) from place_scene."""
        from mycobot_tasks import place_scene as ps
        sa = ps.parse(shlex.split(self.args.scene_args) + self.bot_extra)
        objs, info = ps.build(sa)
        c, s, y = objs['desk'][0]
        return np.array(c[:2], float), np.array(s[:2], float), float(y), float(info['desk_top'])

    def on_desk(self, xy, yaw, margin=0.01):
        c, s, dy, _ = self._desk_geo
        L, W = self.args.relay_tray[:2]
        for u in (-L / 2, L / 2):
            for v in (-W / 2, W / 2):
                p = np.asarray(xy, float) + rot_z(yaw)[:2, :2] @ np.array([u, v])
                d = rot_z(-dy)[:2, :2] @ (p - c)
                if abs(d[0]) > s[0] / 2 - margin or abs(d[1]) > s[1] / 2 - margin:
                    return False
        return True

    def desk_spots(self, yaw):
        """Desk spot candidates (handle midpoint xy) for the tray at tool yaw: --relay-spots,
        or (default) a 2 cm grid over the desk from the layout: radius within
        --relay-spot-r, the whole tray on the desk (1 cm in from its edges); the radius
        nearest the middle of that range first."""
        a = self.args
        if a.relay_spots:
            return [tuple(map(float, s)) for s in a.relay_spots if self.on_desk(s, yaw)]
        c, sz, dy, _ = self._desk_geo
        r0, r1 = a.relay_spot_r
        R = rot_z(dy)[:2, :2]
        out = []
        for u in np.arange(-sz[0] / 2, sz[0] / 2 + 1e-9, 0.02):
            for v in np.arange(-sz[1] / 2, sz[1] / 2 + 1e-9, 0.02):
                p = c + R @ np.array([u, v])
                p = np.round(p / 0.02) * 0.02          # on the carry grid
                r = float(np.linalg.norm(p))
                if r0 <= r <= r1 and self.on_desk(p, yaw):
                    out.append((float(p[0]), float(p[1])))
        out = sorted(set(out), key=lambda p: abs(math.hypot(*p) - (r0 + r1) / 2))
        return out

    # ------------------------------------------------------------------ IK
    def _ik_v(self, p, yaw, q0):
        """Tool straight down at p, heading yaw, near q0 (joint windows LO..HI) or None."""
        fk, R = self._rl_arm.fk, rot_z(yaw) @ DOWN

        def f(q):
            T = fk(q)
            return np.r_[(T[:3, 3] - p) * 10, (T[:3, :3].T @ R - np.eye(3)).ravel() * 0.3]
        r = least_squares(f, np.clip(q0, LO, HI), bounds=(LO, HI))
        return r.x if np.linalg.norm(r.fun) < 3e-3 else None

    def _ik_t(self, p, yaw, q0, tilt=None):
        """TCP at p, tool z within tilt of straight down, x heading yaw, near q0. Inside
        the limit the tilt is still pulled toward 0 (--relay-tilt-weight): the tray tilts
        only as much as the arm needs there, level wherever it can be."""
        tl = math.radians(self.args.relay_tilt) if tilt is None else tilt
        w = self.args.relay_tilt_weight
        fk = self._rl_arm.fk

        def f(q):
            T = fk(q)
            ang = math.acos(max(-1.0, min(1.0, -T[2, 2])))
            hx = math.atan2(T[1, 0], T[0, 0])
            return np.r_[(T[:3, 3] - p) * 10, max(0.0, ang - tl) * 3,
                         math.remainder(hx - yaw, 2 * math.pi) * 0.5, ang * w,
                         (q - q0) * 0.02]
        r = least_squares(f, np.clip(q0, LO, HI), bounds=(LO, HI))
        return r.x if np.linalg.norm(r.fun[:5]) < 4e-3 else None

    def _ik_any(self, p, yaw, flip, n=40):
        """Vertical IK with J1 within 60 deg of the direction to p (flip: the opposite one,
        reversed arm), collision free with the gripper open (no tray held)."""
        b = math.atan2(p[1], p[0]) + (math.pi if flip else 0.0)
        for _ in range(n):
            self._check()
            q0 = self._rl_rng.uniform(LO, HI)
            q0[0] = math.remainder(b + self._rl_rng.normal(0, 0.3), 2 * math.pi)
            q = self._ik_v(p, yaw, q0)
            if q is not None and abs(math.remainder(q[0] - b, 2 * math.pi)) < math.radians(60) \
                    and not self._contacts(q, held=False):
                return q
        return None

    # ------------------------------------------------------------------ collisions
    def _contacts(self, q, held=True, touch=(), pad=None, at_tray=None, grip=None):
        """Colliding pairs at q (gripper --hold-grip with the tray / --grasp-open without).
        touch: names of the scene objects the tray bottom stands on right now (putting down
        / just lifted, see _touching): tray contacts with exactly those are normal and
        ignored, with anything else not. base-desk is always reported: ignored.
        grip: gripper opening of the empty arm (default --grasp-open)."""
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity
        from sensor_msgs.msg import JointState
        a = self.args
        if pad is None:                       # planning: --relay-plan-pad clearance
            pad = a.relay_plan_pad
        rq = GetStateValidity.Request(group_name='arm')
        rq.robot_state = RobotState(
            joint_state=JointState(name=ARM_JOINTS + ['gripper_controller'],
                                   position=[float(v) for v in q] +
                                   [float(a.hold_grip if held else
                                          (a.grasp_open if grip is None else grip))]),
            attached_collision_objects=[self._tray_attached(pad)] if held else [], is_diff=False)
        r = _wait(self._rl_sv.call_async(rq), 20.0)
        if r is None:
            raise Stop('check_state_validity: no answer')
        out = set()
        for c in r.contacts:
            pr = tuple(sorted((c.contact_body_1, c.contact_body_2)))
            if pr == ('base', 'desk'):
                continue
            if touch and TRAY_ID in pr and (set(pr) - {TRAY_ID}) <= set(touch):
                continue
            # empty fingers at the handle of a tray standing in the world (at_tray = its
            # body top z): the fingers straddle the handle above the body: not a collision
            if at_tray is not None and TRAY_ID in pr and (set(pr) - {TRAY_ID}) <= set(TOUCH) \
                    and self._rl_arm.fk(q)[2, 3] < at_tray + 0.01:
                continue
            out.add(pr)
        return sorted(out)

    def _touching(self, q, surfaces):
        """surfaces [(z, object names)]: -> the names of those with the tray bottom within
        --relay-touch over their z (the tray is put on / taken from them now)."""
        z = self._rl_arm.fk(q)[2, 3] - self._tip_to_bottom()
        return set().union(*[set(n) for s, n in surfaces if z < s + self.args.relay_touch])

    # ------------------------------------------------------------------ path pieces
    def _chain(self, q, pts, yaw, held=True, tilt=False, surfaces=(), label=''):
        """Chained IK along pts (each step <= --max-step-jump), every 2nd point (and the
        last) collision checked. -> joint list or RelayFail."""
        a, out = self.args, []
        for i, p in enumerate(pts):
            self._check()
            qn = self._ik_t(p, yaw, q) if tilt else self._ik_v(p, yaw, q)
            if qn is None:
                raise RelayFail(f'{label}: no IK at {np.round(p * 1000).astype(int)} mm')
            if math.degrees(float(np.max(np.abs(qn - q)))) > a.max_step_jump:
                raise RelayFail(f'{label}: IK branch jump at {np.round(p * 1000).astype(int)} mm')
            if i % 2 == 0 or i == len(pts) - 1:
                c = self._contacts(qn, held, touch=self._touching(qn, surfaces) if surfaces else ())
                if c:
                    raise RelayFail(f'{label}: collision at {np.round(p * 1000).astype(int)} mm '
                                    f'{c[:2]}')
            out.append(qn)
            q = qn
        return out

    def _turn(self, q, p, y0, y1, tilt, label):
        out = []
        n = max(2, int(abs(math.degrees(y1 - y0)) / 3))
        for t in np.linspace(0, 1, n + 1)[1:]:
            self._check()
            y = y0 + (y1 - y0) * t
            qn = self._ik_t(p, y, q) if tilt else self._ik_v(p, y, q)
            if qn is None or math.degrees(float(np.max(np.abs(qn - q)))) > self.args.max_step_jump:
                raise RelayFail(f'{label}: turn stuck at {math.degrees(y):.0f} deg (J6 '
                                f'{math.degrees(q[5]):.0f})')
            out.append(qn)
            q = qn
        c = self._contacts(q)
        if c:
            raise RelayFail(f'{label}: collision after the turn {c[:2]}')
        return out

    def _straighten(self, q, p, yaw, label):
        out, tl = [], math.radians(self.args.relay_tilt)
        for t in np.linspace(0, 1, 21)[1:]:
            qn = self._ik_t(p, yaw, q, tilt=tl * (1 - t) + 1e-3)
            if qn is None or math.degrees(float(np.max(np.abs(qn - q)))) > self.args.max_step_jump:
                raise RelayFail(f'{label}: straightening stuck at {math.degrees(tl * (1 - t)):.0f} '
                                f'deg (J6 {math.degrees(q[5]):.0f})')
            out.append(qn)
            q = qn
        qn = self._ik_v(p, yaw, q)
        if qn is None or math.degrees(float(np.max(np.abs(qn - q)))) > self.args.max_step_jump:
            raise RelayFail(f'{label}: no exact vertical pose')
        if self._contacts(qn):
            raise RelayFail(f'{label}: collision when straightened')
        return out + [qn]

    def _grid(self, S, qS, yaw, goals, label):
        """A* over the --relay-carry-z plane (2 cm grid, tool tilted <= --relay-tilt, tray
        held) from S toward the goals (heuristic: straight distance to the nearest goal not
        reached yet). A generator: yields (goal, joint path) as each goal is reached, so the
        caller stops as soon as one works out; ends when none is left / nothing reachable."""
        a, z = self.args, self.args.relay_carry_z
        x0, x1, y0, y1 = a.relay_grid
        xs = np.round(np.arange(x0, x1 + 1e-9, 0.02), 3)
        ys = np.round(np.arange(y0, y1 + 1e-9, 0.02), 3)
        goals = [tuple(map(float, g)) for g in goals]
        nodes = [(float(x), float(y)) for x in xs for y in ys
                 if math.hypot(x, y) >= a.relay_min_r] + goals
        S = tuple(map(float, S))
        left, done = set(goals), set()

        def h(v):
            return min((math.dist(v, g) for g in left), default=0.0)
        best, qof, par, eq, pq = {S: 0.0}, {S: qS}, {}, {}, [(h(S), S)]
        while pq and left:
            self._check()
            _, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            d = best[u]
            if u in left:
                left.discard(u)
                path, v = [], u
                while v != S:
                    path = eq[v] + path
                    v = par[v]
                print(f'  relay {label}: grid {len(done)} points settled -> goal '
                      f'({u[0]:+.2f}, {u[1]:+.2f})', flush=True)
                yield u, path
                if not left:
                    return
                pq = [(best[v] + h(v), v) for _, v in pq]        # nearest goal changed
                heapq.heapify(pq)
            for v in nodes:
                dv = math.dist(u, v)
                if v in done or dv > 0.03 or dv < 1e-9 or d + dv >= best.get(v, 1e9):
                    continue
                qq, e, ok = qof[u], [], True
                for p in seg([u[0], u[1], z], [v[0], v[1], z]):
                    qn = self._ik_t(p, yaw, qq)
                    if qn is None or math.degrees(float(np.max(np.abs(qn - qq)))) > a.max_step_jump:
                        ok = False
                        break
                    e.append(qn)
                    qq = qn
                # every point of the edge (5 mm apart), not only its end: between two clear
                # grid points the tray edge can still graze the arm base (dense check)
                if not ok or any(self._contacts(qe) for qe in e[::-1]):
                    continue
                best[v], qof[v], par[v], eq[v] = d + dv, qq, u, e
                heapq.heappush(pq, (best[v] + h(v), v))
        print(f'  relay {label}: grid {len(done)} points settled, '
              f'{len(left)}/{len(goals)} goals not reached', flush=True)

    def _ompl(self, qa, qb, label, grip=None):
        """Empty arm qa -> qb (OMPL, gripper open / grip), re-checked every 2 deg."""
        from moveit_msgs.msg import Constraints, JointConstraint, RobotState
        from moveit_msgs.srv import GetMotionPlan
        from sensor_msgs.msg import JointState
        a = self.args
        for k in range(6):
            self._check()
            rq = GetMotionPlan.Request()
            r = rq.motion_plan_request
            r.start_state = RobotState(joint_state=JointState(
                name=ARM_JOINTS + ['gripper_controller'],
                position=[float(v) for v in qa] +
                [float(a.grasp_open if grip is None else grip)]))
            r.group_name, r.num_planning_attempts, r.allowed_planning_time = 'arm', 5, 10.0
            r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = 0.3
            r.goal_constraints = [Constraints(joint_constraints=[
                JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005,
                                tolerance_below=0.005, weight=1.0) for j, v in zip(ARM_JOINTS, qb)])]
            res = _wait(self._rl_plan.call_async(rq), 40.0)
            if res is None or res.motion_plan_response.error_code.val != 1:
                continue
            jt = res.motion_plan_response.trajectory.joint_trajectory
            idx = [list(jt.joint_names).index(j) for j in ARM_JOINTS]
            pts = [np.array(p.positions)[idx] for p in jt.points]
            dense = []
            for u, v in zip(pts, pts[1:]):
                m = max(1, int(math.degrees(float(np.max(np.abs(v - u)))) / 2))
                dense += [u + (v - u) * t for t in np.linspace(0, 1, m + 1)[1:]]
            if any(self._contacts(q, held=False, grip=grip) for q in dense):
                continue
            return dense + [np.asarray(qb, float)]
        raise RelayFail(f'{label}: no collision-free OMPL path')

    # ------------------------------------------------------------------ planning
    def _yaws(self, lr_yaw):
        """Tool yaws that pinch the handle with tray L -> R lr_yaw (fingers either way)."""
        y = lr_yaw + math.radians(self.args.grip_yaw_offset)
        return [math.remainder(y, 2 * math.pi), math.remainder(y + math.pi, 2 * math.pi)]

    def plan_leg_a(self, d11):
        """D11 -> desk spot. d11 = dict(h=handle top xyz, lr=tray L -> R yaw).
        A generator: yields dict(steps, spot, yc, q_up, q_rel) for every desk spot that
        works from D11 (q_up: empty arm over the tray on the desk), spots within 5 cm of
        one in self._rl_bad_spots (desk -> TurtleBot failed there) skipped; RelayFail
        when none is left."""
        a = self.args
        h = np.asarray(d11['h'], float)
        surf = h[2] - a.tray_height
        pg, pa = h - [0, 0, a.grasp_depth], h + [0, 0, a.approach_above]
        desk_tip = self._desk_geo[3] + self._tip_to_bottom()
        zc = a.relay_carry_z
        why = []
        rad = h[:2] / max(1e-6, np.linalg.norm(h[:2]))
        perp = np.array([rad[1], -rad[0]])
        # out of the shelf: 2 cm sideways (either way, or not), then straight toward the base
        # to --relay-pull-r, then up to the carry height
        pulls = [(s, rad * a.relay_pull_r + perp * s) for s in (0.02, -0.02, 0.0)]
        for flip in (True, False):
            for y0 in self._yaws(d11['lr']):
                tag = f"D11 {'reversed' if flip else 'normal'} arm, tool yaw {math.degrees(y0):.0f}"
                qd = self._ik_any(pg, y0, flip)
                if qd is None:
                    why.append(f'{tag}: no grasp pose')
                    continue
                try:
                    up = self._chain(qd, seg(pg, pa), y0, held=False, label=f'{tag}: approach')
                    approach = up[::-1] + [qd]          # from over the handle (start point) down
                    lift = self._chain(qd, seg(pg, pa), y0, surfaces=[(surf, D11_ON)],
                                       label=f'{tag}: lift')
                except RelayFail as e:
                    why.append(str(e))
                    continue
                for side, pu in pulls:
                    p1 = np.array([pa[0] + perp[0] * side, pa[1] + perp[1] * side, pa[2]])
                    P = np.array([pu[0], pu[1], zc])
                    try:
                        pull = self._chain(lift[-1], seg(pa, p1) + seg(p1, [pu[0], pu[1], pa[2]]) +
                                           seg([pu[0], pu[1], pa[2]], P), y0, label=f'{tag}: pull out')
                    except RelayFail as e:
                        why.append(str(e))
                        continue
                    for turn in (90.0, -90.0):
                        yc = y0 + math.radians(turn)
                        try:
                            tq = self._turn(pull[-1], P, y0, yc, False, f'{tag}: turn {turn:+.0f}')
                        except RelayFail as e:
                            why.append(str(e))
                            continue
                        spots = self.desk_spots(yc)
                        if not spots:
                            why.append(f'{tag}: turn {turn:+.0f}: no spot candidate on the desk')
                            continue
                        reached = set()
                        for s, path in self._grid(P[:2], tq[-1], yc, spots, f'{tag} turn {turn:+.0f}'):
                            reached.add(s)
                            if any(math.dist(s, b) < 0.05 for b in self._rl_bad_spots):
                                continue
                            ps_ = np.array([s[0], s[1], zc])
                            pd = np.array([s[0], s[1], desk_tip])
                            try:
                                q = (path or tq)[-1]
                                st = self._straighten(q, ps_, yc, f'spot {s}')
                                down = self._chain(st[-1], seg(ps_, pd, 0.003), yc,
                                                   surfaces=[(self._desk_geo[3], DESK_ON)],
                                                   label=f'spot {s}: down')
                                upe = self._chain(down[-1], seg(pd, ps_), yc, held=False,
                                                  label=f'spot {s}: empty up')
                            except RelayFail as e:
                                why.append(str(e))
                                continue
                            # in empty along the way the tray comes out (fingers closed),
                            # opened over the handle
                            enter = pull[::-1] + [approach[0]]
                            steps = [('traj', enter, ('in over the handle on D11 (fingers closed)',
                                                      'out of the shelf (fingers closed)')),
                                     ('fingers', (CLOSED, a.grasp_open),
                                      ('open over the handle', 'close for the way out')),
                                     ('traj', approach[1:], ('down to the handle on D11', 'up from D11')),
                                     ('grasp', None, ('close: tray from D11', 'open: tray on D11')),
                                     ('traj', lift + pull, ('lift, pull out of the shelf, up',
                                                            'down, into the shelf, put down')),
                                     ('traj', tq, (f'turn the tray {turn:+.0f}', f'turn the tray {-turn:+.0f}')),
                                     ('traj', path, ('carry (tilted) to the desk spot',
                                                         'carry (tilted) to the shelf')),
                                     ('traj', st + down, ('straighten, put down on the desk',
                                                          'lift off the desk, tilt')),
                                     ('release', None, ('open: tray on the desk', 'close: tray from the desk')),
                                     ('traj', upe, ('empty hand up', 'empty hand down to the tray'))]
                            print(f'  relay leg A: {tag}, pull to ({pu[0]:+.3f}, {pu[1]:+.3f}), turn '
                                  f'{turn:+.0f}, desk spot ({s[0]:+.2f}, {s[1]:+.2f})', flush=True)
                            yield dict(steps=steps, spot=np.array(s), yc=yc, q_up=upe[-1],
                                       q_rel=down[-1])
                        why += [f'{tag}: turn {turn:+.0f}: spot {t} not reached tilted'
                                for t in spots if t not in reached]
        raise RelayFail('D11 <-> desk: no way\n    ' + '\n    '.join(why[-12:]))

    def plan_leg_b(self, A, bot, q_from):
        """Desk spot -> TurtleBot box. bot = dict(h=handle top xyz in the box, lr=L -> R).
        q_from: empty arm now (over the desk tray). -> dict(steps)."""
        a = self.args
        s, yc = A['spot'], A['yc']
        desk_tip = self._desk_geo[3] + self._tip_to_bottom()
        zc = a.relay_carry_z
        ps_, pd = np.array([s[0], s[1], zc]), np.array([s[0], s[1], desk_tip])
        h = np.asarray(bot['h'], float)
        surf = h[2] - a.tray_height
        G = (float(h[0]), float(h[1]))
        pput = h - [0, 0, a.grasp_depth - a.place_drop]
        why = []
        for gflip in (math.pi, 0.0):
            yg = math.remainder(yc + gflip, 2 * math.pi)
            tag = f'regrasp {math.degrees(gflip):.0f}'
            qn = self._ik_any(pd, yg, False)
            if qn is None:
                why.append(f'{tag}: no normal grasp at the desk spot')
                continue
            try:
                lift = self._chain(qn, seg(pd, ps_), yg, surfaces=[(self._desk_geo[3], DESK_ON)],
                                   label=f'{tag}: lift')
                upe = self._chain(qn, seg(pd, ps_), yg, held=False, label=f'{tag}: empty')
            except RelayFail as e:
                why.append(str(e))
                continue
            # the box yaw needing the smaller turn first (the other one runs J6 to its limit)
            for yb in sorted(self._yaws(bot['lr']),
                             key=lambda y: abs(math.remainder(y - yg, 2 * math.pi))):
                d = math.remainder(yb - yg, 2 * math.pi)
                tb = f'{tag}, box tool yaw {math.degrees(yb):.0f}'
                try:
                    tq = self._turn(lift[-1], ps_, yg, yg + d, True, f'{tb}: turn {math.degrees(d):+.0f}')
                except RelayFail as e:
                    why.append(str(e))
                    continue
                yb = yg + d
                path = next((p for _, p in self._grid(ps_[:2], tq[-1], yb, [G], tb)), None)
                if path is None:
                    why.append(f'{tb}: the box not reached tilted')
                    continue
                q, ok = (path or tq)[-1], False
                for hs in (0.08, 0.05, 0.02):
                    pz = h + [0, 0, hs]
                    try:
                        dsc = self._chain(q, seg([G[0], G[1], zc], pz), yb, tilt=True,
                                          label=f'{tb}: tilted descent to +{hs * 100:.0f} cm')
                        st = self._straighten(dsc[-1], pz, yb, f'{tb}: +{hs * 100:.0f} cm')
                        down = self._chain(st[-1], seg(pz, pput, 0.003), yb, surfaces=[(surf, BOX_ON)],
                                           label=f'{tb}: into the box')
                        out = self._chain(down[-1], seg(pput, h + [0, 0, a.approach_above]), yb,
                                          held=False, label=f'{tb}: up out of the box')
                        ok = True
                        break
                    except RelayFail as e:
                        why.append(str(e))
                if not ok:
                    continue
                # the empty arm switch with the tray on the desk as an obstacle
                self._scene(world=[self._tray_world(A['q_rel'], TMP_ID)])
                try:
                    sw = self._ompl(q_from, upe[-1], 'arm switch over the desk tray')
                except RelayFail as e:
                    why.append(str(e))
                    continue
                finally:
                    self._remove_world(TMP_ID)
                steps = [('traj', sw, ('arm switch (empty, over the tray on the desk)',
                                       'arm switch back (empty, over the tray on the desk)')),
                         ('traj', upe[::-1][1:] + [qn], (f'down to the tray (fingers turned '
                                                         f'{math.degrees(gflip):.0f})', 'empty hand up')),
                         ('grasp', None, ('close: tray from the desk', 'open: tray on the desk')),
                         ('traj', lift, ('lift off the desk', 'put down on the desk')),
                         ('traj', tq, (f'turn the tray {math.degrees(d):+.0f}',
                                       f'turn the tray {-math.degrees(d):+.0f}')),
                         ('traj', path, ('carry (tilted) over the TurtleBot', 'carry (tilted) to the desk spot')),
                         ('traj', dsc + st + down, ('lower tilted, straighten, into the box',
                                                    'lift out of the box, tilt, up')),
                         ('release', None, ('open: tray in the box', 'close: tray from the box')),
                         ('traj', out, ('empty hand up', 'down to the handle in the box'))]
                print(f'  relay leg B: {tb}, straightened at +{hs * 100:.0f} cm', flush=True)
                return dict(steps=steps, yb=yb)
        raise RelayFail('desk <-> TurtleBot: no way\n    ' + '\n    '.join(why[-12:]))

    def relay_verify(self, steps, title, start_tray=True):
        """Every planned path checked again densely (joints <= 1 deg apart), as it will run:
        the tray attached while held, standing in the world where it was put down (and,
        start_tray, where it stands before the first grasp). Once as planned (any contact
        -> Stop) and once with the tray grown by --relay-pad (contacts there = less than
        that clearance: reported). -> list of the close spots."""
        a = self.args
        H = a.relay_tray[2]
        t0 = time.monotonic()
        first_grasp = next(i for i, (k, *_) in enumerate(steps) if k == 'grasp')
        q_g = next(d[-1] for k, d, _ in reversed(steps[:first_grasp]) if k == 'traj' and d)
        res = {}
        # the empty gripper opening at the start: before the first fingers step (no tray
        # taken / put before it), else --grasp-open
        fw0 = a.grasp_open
        for k, d, _ in steps:
            if k in ('grasp', 'release'):
                break
            if k == 'fingers':
                fw0 = d[0]
                break
        for pad in (0.0, a.relay_pad):
            held, prev, bad, fw = False, None, [], fw0
            tray_top = None
            if start_tray:
                self._scene(world=[self._tray_world(q_g, TRAY_ID, pad)])
                tray_top = self._rl_arm.fk(q_g)[2, 3] - self._tip_to_bottom() + H
            try:
                for kind, data, (label, _) in steps:
                    self._check()
                    if kind == 'grasp':
                        held, tray_top = True, None
                        self._remove_world(TRAY_ID)
                        continue
                    if kind == 'fingers':
                        fw = data[1]
                        continue
                    if kind == 'release':
                        held, fw = False, a.grasp_open
                        self._scene(world=[self._tray_world(prev, TRAY_ID, pad)])
                        tray_top = self._rl_arm.fk(prev)[2, 3] - self._tip_to_bottom() + H
                        continue
                    if not data:
                        continue
                    pts = ([prev] if prev is not None else []) + list(data)
                    dense = [pts[0]]
                    for u, v in zip(pts, pts[1:]):
                        m = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(v - u)))))))
                        dense += [u + (v - u) * t for t in np.linspace(0, 1, m + 1)[1:]]
                    hit = {}
                    for q in dense:
                        touch = self._touching(q, self._rl_surfaces) if held else ()
                        for pr in self._contacts(q, held, touch, pad,
                                                 None if held else tray_top, grip=fw):
                            hit[pr] = hit.get(pr, 0) + 1
                    if hit:
                        bad.append((label, len(dense), hit))
                    prev = data[-1]
            finally:
                self._remove_world(TRAY_ID)
            res[pad] = bad
        if res[0.0]:
            raise Stop(f'{title}: COLLISION in the dense check (nothing moved):\n' + '\n'.join(
                f'  {lb}: {sum(h.values())} of {n} poses, {sorted(h)[:3]}' for lb, n, h in res[0.0]))
        near = res[a.relay_pad]
        self.say(f'{title}: dense collision check OK ({time.monotonic() - t0:.0f} s)' +
                 ('' if not near else f'; closer than {a.relay_pad * 1000:.0f} mm in '
                  f'{len(near)} paths:'))
        for lb, n, h in near:
            print(f'    {lb}: {sum(h.values())} of {n} poses: '
                  + ', '.join(f'{p[0]}-{p[1]} x{c}' for p, c in sorted(h.items())[:4]), flush=True)
        return near

    def relay_start_tray(self, steps):
        """The tray where it stands before the first grasp into the scene (so the move to
        the start goes around it; the grasp takes it out again)."""
        i = next(i for i, (k, *_) in enumerate(steps) if k == 'grasp')
        q_g = next(d[-1] for k, d, _ in reversed(steps[:i]) if k == 'traj' and d)
        if self.args.go:
            self._scene(world=[self._tray_world(q_g)])

    @staticmethod
    def relay_reverse(steps):
        """The same motion backwards: paths reversed, grasp <-> release. A forward path
        holds the points after its start (= the end of the path before), so backwards it
        drops its own last point (where the arm is) and ends at that start."""
        prev, start = None, {}
        for i, (kind, data, _) in enumerate(steps):
            if kind == 'traj' and data:
                start[i] = prev
                prev = data[-1]
        out = []
        for i in reversed(range(len(steps))):
            kind, data, (lf, lb) = steps[i]
            if kind == 'traj':
                if not data:
                    continue
                back = list(reversed(data))[1:] + ([start[i]] if start[i] is not None else [])
                out.append(('traj', back, (lb, lf)))
            elif kind == 'fingers':
                out.append(('fingers', (data[1], data[0]), (lb, lf)))
            else:
                out.append(({'grasp': 'release', 'release': 'grasp'}[kind], None, (lb, lf)))
        return out

    def _times(self, pts):
        """Durations of the steps between joint points pts: smooth start / stop along the
        path. Path metric per step = finger tip travel + tool rotation x --relay-lever (a
        tray edge); speed <= --relay-tcp-speed and every joint <= --relay-speed, speeding
        up / slowing down at <= --relay-accel (forward / backward pass), 0 at both ends."""
        a = self.args
        T = [self._rl_arm.fk(q) for q in pts]
        n = len(pts) - 1
        if n < 1:
            return []
        m, lim = [], []
        for i in range(n):
            d = float(np.linalg.norm(T[i + 1][:3, 3] - T[i][:3, 3]))
            R = T[i][:3, :3].T @ T[i + 1][:3, :3]
            ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
            mi = max(d + ang * a.relay_lever, 1e-5)
            J = float(np.max(np.abs(np.asarray(pts[i + 1]) - np.asarray(pts[i]))))
            m.append(mi)
            lim.append(min(a.relay_tcp_speed, a.relay_speed * mi / J if J > 1e-9 else 1e9))
        v = [0.0] * (n + 1)
        for i in range(n):                                  # speeding up
            cap = min(lim[i], lim[i + 1]) if i + 1 < n else 0.0
            v[i + 1] = min(cap, math.sqrt(v[i] ** 2 + 2 * a.relay_accel * m[i]))
        for i in reversed(range(n)):                        # slowing down
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a.relay_accel * m[i]))
        out = []
        for i in range(n):
            s = v[i] + v[i + 1]
            out.append(max(0.01, 2 * m[i] / s if s > 1e-6 else 2 * math.sqrt(m[i] / a.relay_accel)))
        return out

    def relay_summary(self, steps, title):
        """Steps with their duration (as they will run), most tray tilt and fastest finger
        tip speed: the numbers for 'does anything slide out of the tray'."""
        qs = [q for k, d, _ in steps if k == 'traj' for q in d]
        j6 = np.degrees([q[5] for q in qs])
        tot, worst = 0.0, 0.0
        lines = []
        held = False
        for k, d, (lf, _) in steps:
            if k != 'traj':
                if k in ('grasp', 'release'):
                    held = k == 'grasp'
                lines.append(f'    {k:7s} {lf}')
                continue
            if not d:
                continue
            dts = self._times(list(d))
            Ts = [self._rl_arm.fk(q) for q in d]
            tilt = max(math.degrees(math.acos(max(-1.0, min(1.0, -T[2, 2])))) for T in Ts)
            sp = max([float(np.linalg.norm(Ts[i + 1][:3, 3] - Ts[i][:3, 3])) / dt
                      for i, dt in enumerate(dts)] or [0.0])
            tot += sum(dts)
            if held:
                worst = max(worst, tilt)
            lines.append(f'    traj    {lf} ({len(d)} pts, {sum(dts):.1f} s, tilt <= {tilt:.1f} deg, '
                         f'tip <= {sp * 1000:.0f} mm/s){"" if held else "  [empty]"}')
        self.say(f'{title}: {sum(k == "traj" for k, *_ in steps)} paths, {len(qs)} points, about '
                 f'{tot:.0f} s, J6 {j6.min():.0f}..{j6.max():.0f} deg, tray tilt <= {worst:.1f} deg, '
                 f'accel <= {self.args.relay_accel:.2f} m/s2')
        for ln in lines:
            print(ln, flush=True)

    # ------------------------------------------------------------------ running
    def _execute(self, qs, label, expect=None):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        a = self.args
        self.wait_still()
        q0 = self.node.q()
        if q0 is None:
            raise Stop('no joint_states')
        # expect: where the path before ended (the planned start of this one); qs[0] is
        # already the first step away from it
        ref = qs[0] if expect is None else expect
        off = math.degrees(float(np.max(np.abs(q0 - ref))))
        if off > 3.0:
            raise Stop(f'{label}: the arm is {off:.1f} deg away from the planned start')
        pts = [q0] + list(qs)
        dts = self._times(pts)
        jt, t = JointTrajectory(joint_names=list(ARM_JOINTS)), 0.0
        for v, dt in zip(pts, [0.0] + dts):
            t += dt
            jt.points.append(JointTrajectoryPoint(
                positions=[float(x) for x in v],
                time_from_start=Duration(sec=int(t), nanosec=int((t % 1.0) * 1e9))))
        self.say(f'moving: {label} ({len(qs)} points, {t:.1f} s)')
        ok, msg = self.mv.execute(jt)
        self._check()
        if not ok:
            raise Stop(f'{label}: execution failed ({msg})')

    def relay_run(self, steps):
        """Run planned relay steps (one confirm per step)."""
        a = self.args
        expect = None
        for kind, data, (label, _) in steps:
            self._check()
            if kind == 'traj':
                if not data:
                    continue
                self.confirm(label)
                if a.go:
                    self._execute(data, label, expect)
                else:
                    self.say(f'(dry run) {label}')
                expect = data[-1]
            elif kind == 'grasp':
                self.confirm(label)
                ok, txt = self.grasp_check(self.gripper(False))
                self.say(txt + ('' if ok else ' : MISSED'))
                if not ok:
                    self.gripper(True, a.grasp_open)
                    raise Stop(f'{label}: nothing between the fingers (gripper open again)')
                self.holding = True
                if a.go:
                    self._attach()
            elif kind == 'fingers':
                self.confirm(label)
                self._fingers(data[1])
            elif kind == 'release':
                self.confirm(label)
                self.gripper(True, a.grasp_open)
                self.holding, self.grip = False, None
                if a.go:
                    self._detach(self.node.q())

    def _fingers(self, w):
        """Empty gripper to opening w (CLOSED = closed)."""
        if w <= CLOSED + 1e-6:
            self.gripper(False)
        else:
            self.gripper(True, w)

    def relay_go_to(self, q, label, grip=None):
        """From wherever the arm is to the first planned point: OMPL (empty arm, up to 6
        tries, every 2 deg re-checked: the start over D11 is a narrow gap between the rack
        and the mast where one OMPL path often fails MoveIt's own path check).
        grip: the gripper opening on the way (default --grasp-open)."""
        grip = self.args.grasp_open if grip is None else grip
        self._fingers(grip)
        self.wait_still()
        q0 = self.node.q()
        if q0 is None:
            raise Stop('no joint_states')
        if math.degrees(float(np.max(np.abs(q0 - q)))) < 1.0:
            return
        try:
            path = self._ompl(q0, q, label, grip=grip)
        except RelayFail as e:
            raise Stop(f'{e} (from the arm pose now)')
        self._execute(path, label, expect=q0)

    # ------------------------------------------------------------------ poses
    def relay_d11_file(self):
        """D11 tray pose from the layout export (--relay-d11): dict(h, lr)."""
        import yaml
        with open(os.path.expanduser(self.args.relay_d11)) as f:
            d = yaml.safe_load(f)
        return dict(h=np.array(d['handle_xyz'], float), lr=math.radians(d['yaw_lr_deg']))

    def relay_bot_place(self):
        """TurtleBot place (self.bot from the marker): dict(h, lr)."""
        b = self.bot
        return dict(h=np.array([b['place_xy'][0], b['place_xy'][1],
                                b['place_z'] + self.args.tray_height], float),
                    lr=float(b['place_yaw']))

    def relay_layout(self):
        """--layout-poses (simulation): the poses as arranged in RViz (ros2 run mycobot_tasks
        layout writes them on every change), read now: no camera, no restart needed.
        -> dict(d11, bot_place, bot_tray); self.bot from the layout marker."""
        import yaml
        if self.conn not in ('fake', None):
            raise Stop(f'--layout-poses is for the simulation (driver {self.conn!r}): the real '
                       'run measures the TurtleBot / tray with the camera')

        def load(k):
            with open(os.path.join(os.path.expanduser(self.args.layout_dir), f'layout_{k}.yaml')) as f:
                return yaml.safe_load(f)
        b, tb, d = load('bot'), load('tray_bot'), load('tray')
        self.say(f"layout poses ({b['stamp']}): no camera")
        self.bot = self.bot_from_marker(b['marker_xyz'], math.radians(b['yaw_deg']))
        return dict(d11=dict(h=np.array(d['handle_xyz'], float), lr=math.radians(d['yaw_lr_deg'])),
                    bot_place=self.relay_bot_place(),
                    # the tray in the box as n leaves it: marker +x + --bot-yaw-offset
                    bot_tray=dict(h=np.array(tb['handle_xyz'], float),
                                  lr=math.radians(tb['yaw_lr_deg'] + self.args.bot_yaw_offset)))

    def _relay_key(self, *poses):
        """Cache key: the poses (1 mm / 0.5 deg) + scene + relay options."""
        a = self.args
        lay = os.path.expanduser('~/.ros/layout.yaml')
        return (tuple(np.round(np.concatenate([p['h'] for p in poses]) * 1000)),
                tuple(round(math.degrees(p['lr']) * 2) for p in poses),
                a.scene_args, tuple(self.bot_extra),
                os.path.getmtime(lay) if '--layout' in a.scene_args and os.path.exists(lay) else 0,
                tuple(a.relay_tray), a.relay_tilt, a.relay_carry_z, tuple(a.relay_spots or ()),
                tuple(a.relay_spot_r), a.relay_plan_pad)

    def relay_plan_a(self, d11):
        """n, first part: D11 -> desk spot only (the TurtleBot is measured after the tray is
        on the desk). The first desk spot that works from D11. -> A (see plan_leg_a)."""
        self.relay_setup()
        self._desk_geo = self._desk()
        self.relay_clear()
        a = self.args
        self._rl_surfaces = [(float(d11['h'][2]) - a.tray_height, D11_ON),
                             (self._desk_geo[3], DESK_ON)]
        key = ('A',) + self._relay_key(d11)
        cached = getattr(self, '_rl_cache_a', None)
        if cached is not None and cached[0] == key:
            self.say('relay plan D11 -> desk: same pose and scene as the last plan, reused')
            return cached[1]
        t0 = time.monotonic()
        self._rl_bad_spots = []
        try:
            for A in self.plan_leg_a(d11):
                why = self._regrasp_ok(A)
                if why is None:
                    break
                sp = tuple(A['spot'])
                self._rl_bad_spots.append(sp)
                self.say(f'relay: desk spot ({sp[0]:+.2f}, {sp[1]:+.2f}) works from D11 but '
                         f'{why}: next spot')
                if len(self._rl_bad_spots) >= a.relay_max_spots:
                    raise RelayFail(f'{len(self._rl_bad_spots)} desk spots tried '
                                    '(--relay-max-spots)')
        except RelayFail as e:
            raise Stop(f'relay plan D11 -> desk failed:\n  {e}')
        self._rl_cache_a = (key, A)
        self.say(f'relay plan D11 -> desk OK in {time.monotonic() - t0:.0f} s: desk spot '
                 f'({A["spot"][0]:+.2f}, {A["spot"][1]:+.2f})')
        return A

    def _regrasp_ok(self, A):
        """Quick check of a desk spot before the TurtleBot is known: the normal arm can take
        the tray there with the fingers turned 180 deg and lift it to the carry height (the
        regrasp the box needs: study notes 2026-11-23; 0 deg runs J6 to its limit at the
        box). -> None or why not."""
        a = self.args
        s, yc = A['spot'], A['yc']
        pd = np.array([s[0], s[1], self._desk_geo[3] + self._tip_to_bottom()])
        ps_ = np.array([s[0], s[1], a.relay_carry_z])
        yg = math.remainder(yc + math.pi, 2 * math.pi)
        qn = self._ik_any(pd, yg, False)
        if qn is None:
            return 'no normal-arm grasp there (fingers turned 180)'
        try:
            self._chain(qn, seg(pd, ps_), yg, surfaces=[(self._desk_geo[3], DESK_ON)],
                        label='regrasp 180: lift')
        except RelayFail as e:
            return str(e)
        return None

    def relay_plan_b(self, A, bot, q_from):
        """n, second part: desk spot (tray standing there, arm empty at q_from) ->
        TurtleBot box. -> B (see plan_leg_b) or Stop."""
        a = self.args
        self._rl_surfaces = [s for s in self._rl_surfaces if s[1] != BOX_ON] + \
            [(float(bot['h'][2]) - a.tray_height, BOX_ON)]
        # the tray standing on the desk (world object since the release) is the one to be
        # held here: planned without it (plan_leg_b puts it in as an obstacle for the empty
        # arm switch only); the caller puts it back
        self._remove_world(TRAY_ID)
        t0 = time.monotonic()
        try:
            B = self.plan_leg_b(A, bot, q_from)
        except RelayFail as e:
            raise Stop(f'relay plan desk -> TurtleBot failed:\n  {e}')
        self.say(f'relay plan desk -> TurtleBot OK in {time.monotonic() - t0:.0f} s')
        return B

    def relay_plan(self, d11, bot):
        """Both legs; the same poses (1 mm / 0.5 deg) and scene as the last plan -> that
        plan again (e.g. a dry run first, then --go / auto)."""
        self.relay_setup()
        self._desk_geo = self._desk()
        self.relay_clear()
        a = self.args
        lay = os.path.expanduser('~/.ros/layout.yaml')
        key = (tuple(np.round(np.r_[d11['h'], bot['h']] * 1000)),
               round(math.degrees(d11['lr']) * 2), round(math.degrees(bot['lr']) * 2),
               a.scene_args, tuple(self.bot_extra),
               os.path.getmtime(lay) if '--layout' in a.scene_args and os.path.exists(lay) else 0,
               tuple(a.relay_tray), a.relay_tilt, a.relay_carry_z, tuple(a.relay_spots or ()),
               tuple(a.relay_spot_r), a.relay_plan_pad)
        self._rl_surfaces = [(float(d11['h'][2]) - a.tray_height, D11_ON),
                             (self._desk_geo[3], DESK_ON),
                             (float(bot['h'][2]) - a.tray_height, BOX_ON)]
        cached = getattr(self, '_rl_cache', None)
        if cached is not None and cached[0] == key:
            self.say('relay plan: same poses and scene as the last plan, reused')
            return cached[1], cached[2]
        t0 = time.monotonic()
        # desk spots in turn until the desk -> TurtleBot part works from one of them
        self._rl_bad_spots, whyb, B = [], [], None
        try:
            for A in self.plan_leg_a(d11):
                try:
                    B = self.plan_leg_b(A, bot, A['q_up'])
                    break
                except RelayFail as e:
                    sp = tuple(A['spot'])
                    self._rl_bad_spots.append(sp)
                    whyb.append(f'desk spot ({sp[0]:+.2f}, {sp[1]:+.2f}): {e}')
                    self.say(f'relay: desk spot ({sp[0]:+.2f}, {sp[1]:+.2f}) works from D11, '
                             f'not to the TurtleBot: next spot')
                    if len(self._rl_bad_spots) >= a.relay_max_spots:
                        raise RelayFail(f'{len(whyb)} desk spots tried (--relay-max-spots)')
        except RelayFail as e:
            raise Stop('relay plan failed, nothing moved:\n  ' + '\n  '.join(whyb[-3:] + [str(e)]))
        if B is None:
            raise Stop('relay plan failed, nothing moved:\n  ' + '\n  '.join(whyb[-3:]))
        self._rl_cache = (key, A, B)
        self.say(f'relay plan OK in {time.monotonic() - t0:.0f} s: desk spot '
                 f'({A["spot"][0]:+.2f}, {A["spot"][1]:+.2f})')
        return A, B
