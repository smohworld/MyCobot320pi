"""HandleApp: the tray_handle_pick actions (pick, pick + place, put back) on top of the
detection / scan / TurtleBot / planning mixins."""
import math
import os
import threading
import time

import numpy as np

from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import App, Stop
from mycobot_tasks import regions as rg
from mycobot_tasks.tray_handle import ID_OFFSET, MarkerPose
from mycobot_tasks.handle_pick.base import BaseApp
from mycobot_tasks.handle_pick.bot import BotMixin
from mycobot_tasks.handle_pick.detect import DetectMixin
from mycobot_tasks.handle_pick.plan import PlanMixin
from mycobot_tasks.handle_pick.scan import ScanMixin


class HandleApp(DetectMixin, ScanMixin, BotMixin, PlanMixin, BaseApp):
    def __init__(self, node, args):
        self.pin = None                   # dict(p=handle top midpoint xyz, yaw L -> R)
        self.grip = None                  # held tray: dict(pin=xy, tcp_yaw, tray_yaw, heights)
        self.tray = None
        self.prompt = None
        self._go_on = threading.Event()
        self.auto = args.auto
        self.spacing = None               # L-R centre spacing measured at the detection
        self.last = None                  # (time, markers found) for the window
        self.use_scene = args.scene       # place_scene desk / floor instead of the big table
        self.bot = None                   # TurtleBot marker: dict(p, yaw, place_xy, place_z, ...)
        self.bot_extra = []               # place_scene options from the TurtleBot detection
        self.table_from_region = args.table_z is None
        if self.table_from_region:        # tray surface = the shelf region's z (RViz) or desk
            r = rg.saved('shelf', os.path.expanduser(args.regions))
            args.table_z = r['z'] if r else -0.007
            print(f'--table-z {args.table_z:.3f} from '
                  + ('the shelf region (regions.yaml)' if r else 'the default (desk top)'),
                  flush=True)
        App.__init__(self, node, args)
        if self.use_scene:
            self.apply_scene()
        self.replay_q = None
        if args.replay:
            if self.conn not in ('fake', None):
                raise SystemExit(f'--replay is for the simulation; the driver is {self.conn!r}')
            self.load_replay()
        self.mp = MarkerPose(self.K, self.D, args.marker_size)
        self.bot_mp = MarkerPose(self.K, self.D, args.bot_marker_size)
        self.id_l = ID_OFFSET + 2 * args.tray
        self.id_r = self.id_l + 1
        self._settle_now = None           # driver settle_mode last set (driver_settle)
        self._settle_cli = None

    def apply_scene(self):
        """place_scene (desk, legs, robot supports, floor [+ TurtleBot / shelf from
        --scene-args, + the TurtleBot where its marker was found]) into MoveIt and
        the 1.5 x 1.5 m 'calibration_table' out (it blocks everything below the desk)."""
        import shlex
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from mycobot_calibration.moveit_client import _wait
        from mycobot_tasks import place_scene as ps
        sa = ps.parse(shlex.split(self.args.scene_args) + self.bot_extra)
        objs, _ = ps.build(sa)
        self.floor_z = sa.table_z - sa.desk_height

        def send(cos):
            req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
            req.scene.world.collision_objects = cos
            res = _wait(self.mv.scene.call_async(req), 5.0)
            return res is not None and res.success
        for n in ('calibration_table',) + ps.NAMES:      # one by one: missing ones fail
            co = CollisionObject()
            co.header.frame_id, co.id, co.operation = 'base', n, CollisionObject.REMOVE
            send([co])
        if not send(ps.collision_objects(objs, 'base')):
            raise SystemExit('apply_planning_scene failed (place_scene objects)')
        print(f'planning scene: {", ".join(objs)} from place_scene '
              f'[{" ".join([self.args.scene_args] + self.bot_extra).strip() or "defaults"}] '
              '(calibration_table removed)', flush=True)

    def set_table(self, margin=True):
        if not self.use_scene:
            super().set_table(margin)

    def driver_settle(self, mode):
        """Driver settle_mode (precise / once / fast) for the next trajectories; only sent
        when it changes. A driver without the parameter: warned once, motions stay precise."""
        if mode == self._settle_now:
            return
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
        if self._settle_cli is None:
            self._settle_cli = self.node.create_client(
                SetParameters, f'/{self.args.driver_node}/set_parameters')
        why = None
        if not self._settle_cli.wait_for_service(timeout_sec=0.5):
            why = 'no parameter service'
        else:
            req = SetParameters.Request(parameters=[Parameter(
                name='settle_mode', value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                                         string_value=mode))])
            fut = self._settle_cli.call_async(req)
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < 1.0:
                time.sleep(0.005)
            res = fut.result() if fut.done() else None
            if res is None:
                why = 'no answer'
            elif not res.results or not res.results[0].successful:
                why = res.results[0].reason if res.results else 'refused'
        if why:
            if self._settle_now != 'unavailable':
                self.say(f'driver settle_mode {mode}: {why} (old driver?): every motion settles '
                         'precisely')
            self._settle_now = 'unavailable'
            return
        self._settle_now = mode

    def move(self, name, q, slow=False, velocity=None, line=False, mid=False):
        """App.move with the Pilz planners (--planner pilz): line=True -> LIN (straight TCP
        line, tool orientation kept: descents, lifts, carrying the tray), else PTP (straight
        in joint space). --planner ompl = the old behaviour (OMPL, any path).
        mid: an intermediate motion (no grasp / put down / camera view at its end): the
        driver ends it per --mid-settle; others settle precisely."""
        a = self.args
        if a.go and a.mid_settle != 'precise' and self._settle_now != 'unavailable':
            self.driver_settle(a.mid_settle if mid else 'precise')
        if a.planner == 'ompl':
            return super().move(name, q, slow=slow, velocity=velocity)
        self._check()
        if not a.go:
            self.say(f'(dry run) {name}')
            return
        kind = 'LIN' if line else 'PTP'
        self.say(f'moving: {name} ({kind})')
        v = velocity or (a.slow_velocity if slow else a.velocity)
        acc = a.accel
        if line:
            v = a.lin_velocity if velocity is None else velocity
            acc = a.lin_accel
        ok, msg = self.mv.move_joints(ARM_JOINTS, q, v, acc, planner=kind)
        if not ok and line and msg == 'PLANNING_FAILED':
            # near a stretched arm a constant TCP speed needs fast joints: Pilz refuses
            # (joint velocity / acceleration limits) -> slower once more
            self._check()
            self.say(f'{name}: LIN refused at this speed, again slower')
            ok, msg = self.mv.move_joints(ARM_JOINTS, q, v / 3, acc / 4, planner=kind)
        self._check()
        if not ok and not line and not self.holding and msg in ('INVALID_MOTION_PLAN',
                                                                 'PLANNING_FAILED'):
            # the straight joint-space PTP path hits something (scene objects, the gripper
            # into link2): plan around it (OMPL); not with the tray (not in the scene)
            self.say(f'{name}: PTP path blocked ({msg}), planning around it (OMPL)')
            ok, msg = self.mv.move_joints(ARM_JOINTS, q, v, acc)
            self._check()
        if not ok:
            q_now = self.node.q()
            err = np.degrees(np.max(np.abs(q_now - q))) if q_now is not None else np.inf
            if err > 3.0:
                raise Stop(f'{name}: {kind} motion failed ({msg}), joints {err:.1f} deg off')
        time.sleep(0.1)

    def grasp_check(self, r):
        """Gripper close result -> (holding?, text). Decided by the remaining finger gap,
        not by the driver's 'stalled': a thin handle lets the fingers close to within the
        driver's tolerance, which it reports as 'target reached' (= no object)."""
        a = self.args
        if r is None:
            return True, 'dry run'
        if a.replay:
            return True, 'replay (simulation): grasp assumed, no real handle between the fingers'
        g0 = r.position / a.open_position * a.full_open_mm       # joint 1.0 = 100 mm
        # the driver answers as soon as the fingers are within its tolerance (5 of 100) of
        # the target, still closing: read the settled gap from joint_states
        time.sleep(a.grip_settle)
        js = self.node.js.get('gripper_controller')
        gap = g0 if js is None else js / a.open_position * a.full_open_mm
        txt = (f'gripper closed: gap {gap:.1f} mm after {a.grip_settle:.1f} s (at the '
               f'driver result {g0:.1f} mm, stalled {r.stalled}, reached {r.reached_goal})')
        return gap >= a.min_grip_gap, txt

    def act_detect(self):
        """d: detect the tray at any height (detect_any), nothing moves after."""
        if self.holding:
            raise Stop('holding the tray: put it down first')
        keep = self.args.table_z
        try:
            self.detect_any()
        finally:
            self.args.table_z = keep

    def act_pick(self):
        """p: detect the tray at any height (detect_any) and pick it there."""
        a = self.args
        if self.holding:
            raise Stop('already holding the tray')
        keep = a.table_z
        try:
            pin = self.detect_any()
            plan = self.plan_pick(pin['p'][:2], pin['yaw'] + math.radians(a.grip_yaw_offset))
            self.say(f'pick plan OK: above / approach / grasp / lift, tool yaw '
                     f'{math.degrees(plan["tcp_yaw"]):.0f} deg, surface z '
                     f'{a.table_z * 1000:.0f} mm')
            self.do_pick(pin, plan)
        finally:
            a.table_z = keep

    def act_hover(self):
        """v: open + to the planned first pose over the handle (no servo, no grasp)."""
        a = self.args
        if self.holding:
            raise Stop('holding the tray: b = put it back first')
        pin = self.detect_pin()
        xy = pin['p'][:2]
        plan = self.plan_pick(xy, pin['yaw'] + math.radians(a.grip_yaw_offset))
        p, q = plan['legs']['above']
        self.target = p
        self.confirm(f'open + above the handle ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm')
        opened = self.gripper_async(True, self.args.grasp_open)
        self.move('above handle', q)
        opened()
        self.tip_report(p)
        self.say(f'above the handle: finger tips {a.servo_above * 1000:.0f} mm over its top')

    def place_target(self, xy):
        a = self.args
        if a.place_at == 'bot':
            return np.array(self.bot['place_xy'], float)
        if a.place_at == 'shelf':
            return self.shelf_spot()[0]
        if a.place_offset:
            return np.asarray(xy[:2], float) + np.array(a.place_offset[:2], float)
        return np.array(a.place_at[:2], float)

    def act_pick_place(self):
        """Detect, plan everything (pick, transfer, place at --place-offset from the pick
        spot or --place-at), then: pick -> carry -> put down -> open -> up; the arm stays
        there (no home). Nothing moves when one leg has no plan."""
        a = self.args
        if self.holding:
            raise Stop('already holding the tray')
        home = self.home
        if self.wants_bot():             # TurtleBot first (the camera comes back after)
            if self.bot is None or not a.bot_reuse:     # every m: where is it now?
                self.find_bot()
                # default: the tray search starts right from the TurtleBot view (nothing
                # seen there -> shelf region scan); --bot-home: back to home first
                if a.go and a.bot_home and home is not None:
                    self.move('back to home', home, mid=True)
        if a.place_z_spec == 'auto':
            a.place_z = self.bot['place_z']
        if a.place_yaw_spec == 'bot':
            a.place_yaw = math.degrees(self.bot['place_yaw'])
        if a.place_at == 'shelf':         # the shelf region's surface
            a.place_z = self.shelf_spot()[1]
            if a.shelf_yaw is not None:
                a.place_yaw = a.shelf_yaw
        pin = self.detect_pin()
        xy = pin['p'][:2].copy()
        if self.wants_bot() and self.bot is not None and \
                np.linalg.norm(xy - self.bot['place_xy']) < a.on_bot_r and \
                abs(pin['p'][2] - (self.bot['place_z'] + a.tray_height)) < 0.03:
            raise Stop(f'the tray found at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm is already '
                       'on the TurtleBot: put a tray in the tray area (or r = put it back)')
        dst = self.place_target(xy)
        turn = None
        if a.place_yaw is not None:      # tray L -> R direction wanted at the place
            turn = math.remainder(math.radians(a.place_yaw) - pin['yaw'], 2 * math.pi)
        plan = self.plan_pick(xy, pin['yaw'] + math.radians(a.grip_yaw_offset), dst, turn)
        end_yaw = pin['yaw'] + (turn or 0.0)
        self.say(f'pick + place plan OK: ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) -> '
                 f'({dst[0] * 1000:.0f}, {dst[1] * 1000:.0f}) mm, tray L->R '
                 f'{math.degrees(pin["yaw"]):.0f} -> '
                 f'{math.degrees(math.remainder(end_yaw, 2 * math.pi)):.0f} deg, surface z '
                 f'{a.place_z * 1000:.0f} mm')
        # onto the TurtleBot: the next tray is looked for where this one was picked
        self.transfer(pin, plan, dst, remember=not self.wants_bot())
        if self.wants_bot() and a.go:     # surface from the measured handle top
            self.save_origin(pin, float(pin['p'][2]) - a.tray_height)

    def transfer(self, pin, plan, dst, remember=True):
        """Run a planned pick + place: pick -> up -> carry -> put down -> open -> up
        (-> back toward the base); the arm stays there (no home)."""
        a = self.args
        L = plan['legs']
        # grasp -> high is one vertical line (the plan checked grasp -> approach -> above ->
        # high): tried as one LIN, split up only when that fails
        direct = self.do_pick(pin, plan, lift_to=L['high'][1])
        self.confirm('carry: ' + ('' if direct else 'straight up, ') +
                     'across over the place spot (tool yaw kept)')
        if not direct:
            self.move('up', L['above'][1], line=True, mid=True)
            self.move('up high', L['high'][1], line=True, mid=True)
        self.carry(L, plan['carry'])
        self.target = L['p_put'][0]
        self.confirm(f'put down at ({dst[0] * 1000:.0f}, {dst[1] * 1000:.0f}) mm')
        self.move('put down', L['p_put'][1], line=True, velocity=a.grasp_velocity)
        self.tip_report(L['p_put'][0])
        self.confirm('open the gripper (release the tray)')
        self.gripper(True, a.grasp_open)
        self.holding, self.grip = False, None
        if remember:                     # where to look for this tray next time
            self._last_pin = np.array([dst[0], dst[1], a.place_z + a.tray_height])
        self.confirm('lift straight up')
        self.move('lift', L['p_lift'][1], line=True, mid=True)
        if 'p_back' in L:
            self.move('back toward the base', L['p_back'][1], line=True, mid=True)
        self.target = None
        self.say(f'tray placed at ({dst[0] * 1000:.0f}, {dst[1] * 1000:.0f}) mm'
                 + ('' if a.go else ' (dry run)'))

    def carry(self, L, names):
        """high -> via* -> p_high: first as ONE blended Pilz LIN sequence (no stop at the
        via points); when that is refused or breaks off, one LIN per leg as before, from
        the planned point the arm is at / nearest to."""
        a = self.args
        if len(names) > 1 and a.carry_blend > 0 and a.planner == 'pilz':
            if not a.go:
                self.say(f'(dry run) carry: {" -> ".join(names)} in one blended sequence')
                return
            pts = [L['high'][0]] + [L[n][0] for n in names]
            seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(names))]
            # radius at goal i: within both neighbouring segments (Pilz: blend spheres of
            # neighbouring goals must not overlap)
            radii = [min(a.carry_blend, 0.4 * seg[i], 0.4 * seg[i + 1])
                     for i in range(len(names) - 1)] + [0.0]
            if a.mid_settle != 'precise' and self._settle_now != 'unavailable':
                self.driver_settle(a.mid_settle)
            self._check()
            self.say(f'moving: carry {" -> ".join(names)} (one blended LIN sequence, corners '
                     f'{max(radii) * 1000:.0f} mm)')
            q0 = self.node.q()
            start = None if q0 is None else dict(zip(ARM_JOINTS, q0))
            if start is not None and 'gripper_controller' in self.node.js:
                start['gripper_controller'] = self.node.js['gripper_controller']
            ok, msg = self.mv.move_sequence(ARM_JOINTS, [L[n][1] for n in names], radii,
                                            a.lin_velocity, a.lin_accel, start=start)
            self._check()
            if ok:
                time.sleep(0.1)
                return
            # resume per leg after the planned point nearest to the arm now
            q_now = self.node.q()
            chain = ['high'] + list(names)
            rest = list(names)
            if q_now is not None:
                err = [math.degrees(float(np.max(np.abs(q_now - L[n][1])))) for n in chain]
                i = int(np.argmin(err))
                # at a planned point: on from the next one; between points: to the nearest
                rest = chain[i + 1:] if err[i] <= 3.0 else chain[max(1, i):]
            self.say(f'carry sequence failed ({msg}): one LIN per leg instead ('
                     + ' -> '.join(rest) + ')')
            names = rest
        for n in names:
            self.move(f'carry ({n})', L[n][1], line=True, mid=True)

    def lift_direct(self, q):
        """After grasping: one straight LIN up to q (the carry height). -> True when there;
        False when Pilz / the driver refused it (the caller goes up in the planned steps
        from wherever the arm is now). A stop request is passed on."""
        try:
            self.move('lift straight to the carry height (one LIN)', q, line=True, mid=True)
            return True
        except Stop as e:
            if self._abort.is_set():
                raise
            self.say(f'one-LIN lift failed ({e}): going up in steps instead')
            return False

    def do_pick(self, pin, plan, lift_to=None):
        """Run the planned pick legs: open + above -> approach -> grasp -> close -> lift.
        lift_to: joints at the end of the vertical line over the grasp (e.g. 'high'): after
        a good grasp the lift goes there in one LIN first, split up (lift to approach) when
        that fails. -> True when the arm went straight to lift_to."""
        a = self.args
        xy = pin['p'][:2].copy()
        L = plan['legs']
        self.target = L['above'][0]
        self.confirm(f'open + above the handle ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm')
        opened = self.gripper_async(True, self.args.grasp_open)
        self.move('above handle', L['above'][1], mid=True)
        opened()
        self.target = L['grasp'][0]
        self.confirm(f'descend: finger tips {a.grasp_depth * 1000:.0f} mm below the handle top')
        self.move('down to approach', L['approach'][1], slow=True, line=True)
        self.move('grasp', L['grasp'][1], line=True, velocity=a.grasp_velocity)
        self.tip_report(L['grasp'][0])
        self.confirm('close the gripper')
        ok, txt = self.grasp_check(self.gripper(False))
        self.say(txt + ('' if ok else f' < --min-grip-gap {a.min_grip_gap:.1f} mm: MISSED'))
        if ok:
            self.holding = True
            self.grip = dict(pin=xy, tcp_yaw=plan['tcp_yaw'], tray_yaw=pin['yaw'],
                             legs=L, heights=(a.grasp_z, a.approach, a.servo_height))
        else:
            self.gripper(True, a.grasp_open)
        one = ok and lift_to is not None and not a.lift_split
        self.confirm('lift straight up' + (' to the carry height' if one else '')
                     + ('' if ok else ' (MISSED: gripper open again)'))
        direct = one and self.lift_direct(lift_to)
        if not direct:
            self.move('lift', L['approach'][1], line=True, mid=True)
        if not ok:
            raise Stop('nothing between the fingers: gripper open again; d = detect again')
        self.pin = None
        self.say('tray picked' + ('' if a.go else ' (dry run)') +
                 (': lifted level in one LIN to the carry height' if direct else
                  f': lifted level, finger tips '
                  f'{(L["approach"][0][2] - L["grasp"][0][2] - a.grasp_depth) * 1000:.0f} mm '
                  'over the handle top'))
        return direct

    def act_place(self, centre=None):
        g = self.grip
        if centre is None and g is not None and 'legs' in g:
            # put back: the joint values planned with the pick (no new IK)
            a, L = self.args, g['legs']
            self.target = L['put'][0]
            self.confirm('put back: over the pick spot')
            self.move('place above', L['approach'][1], line=True, mid=True)
            self.confirm(f'put down: {a.place_drop * 1000:.0f} mm above the pick height')
            self.move('put down', L['put'][1], line=True, velocity=a.grasp_velocity)
            self.tip_report(L['put'][0])
            self.confirm('open the gripper (release the tray)')
            self.gripper(True, a.grasp_open)
            self.holding, self.grip = False, None
            self.confirm('lift straight up')
            self.move('lift', L['approach'][1], line=True, mid=True)
            self.say('tray put back' + ('' if a.go else ' (dry run)'))
            return
        raise Stop('no put back planned with a pick (started with --holding / restarted): '
                   'put the tray down by hand')
