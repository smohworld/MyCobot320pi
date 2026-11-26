"""HandleApp: the tray_handle_pick actions (n / j / e / r relay, d detect, s search pose)
on top of the detection / scan / TurtleBot / relay mixins."""
import csv
import math
import os
import threading
import time

import numpy as np

from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import App, Stop
from mycobot_tasks import regions as rg
from mycobot_tasks.tray_handle import MarkerPose
from mycobot_tasks.handle_pick.base import BaseApp
from mycobot_tasks.handle_pick.bot import BotMixin
from mycobot_tasks.handle_pick.detect import DetectMixin
from mycobot_tasks.handle_pick.relay import CLOSED, RelayMixin
from mycobot_tasks.handle_pick.scan import ScanMixin


class HandleApp(DetectMixin, ScanMixin, BotMixin, RelayMixin, BaseApp):
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
        # n / j / e / r end pose: fixed (--relay-home, else the arm pose at program start).
        # self.home itself moves with every camera re-centring (look_at), e.g. over the
        # TurtleBot box after r, so the relay would end there instead of at home
        if args.relay_home:
            if len(args.relay_home) != 6:
                raise SystemExit('--relay-home: 6 joint angles J1..J6 [deg]')
            self.relay_home_q = np.radians(np.array(args.relay_home, float))
        else:
            self.relay_home_q = None if self.home is None else np.array(self.home, float)
        if self.use_scene:
            self.apply_scene()
        self.replay_q = None
        if args.replay:
            if self.conn not in ('fake', None):
                raise SystemExit(f'--replay is for the simulation; the driver is {self.conn!r}')
            self.load_replay()
        self.mp = MarkerPose(self.K, self.D, args.marker_size)
        self.bot_mp = MarkerPose(self.K, self.D, args.bot_marker_size)
        from mycobot_tasks.handle_pick import trays as tr
        self.id_l, self.id_r = tr.ids_of(args.tray)
        self.tray_lock = None             # the tray one detection measures (detect_pin)
        self.tray_check = tr.TrayCheck()  # wrong-tray check (later; today all ok)
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
            req.scene.object_colors = ps.object_colors(
                [co.id for co in cos if co.operation == CollisionObject.ADD])
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
        self.mv.label(f'{name} ({kind})', getattr(self, 'track_mode', ''), self.holding)
        v = velocity or (a.slow_velocity if slow else a.velocity)
        acc = a.accel
        if line:
            v = a.lin_velocity if velocity is None else velocity
            acc = a.lin_accel
        k = self.sim_speed()                 # --sim-speed (simulation only)
        v, acc = min(1.0, v * k), min(1.0, acc * k)
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
        if a.replay or a.layout_poses:
            return True, 'simulation: grasp assumed, no real handle between the fingers'
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

    def act_d11_to_bot(self):
        """n: shelf D11 -> TurtleBot through a desk spot (relay.py), in two parts:
        1. camera over D11 (the layout pose --relay-d11), tray handle measured, D11 -> desk
           planned: no plan (or tray not seen) -> the arm back where n started, error;
           else run: the tray on the desk.
        2. TurtleBot marker measured now, desk -> TurtleBot planned and run (no plan: the
           tray stays on the desk, the arm back where n started, error)."""
        self.track_mode = 'n'
        a = self.args
        if self.holding:
            raise Stop('holding the tray: put it down first')
        keep = a.table_z
        q_start = self.node.q()

        def back(why):
            """ERROR, the arm (empty) back to home (else where n started), then the error."""
            self.relay_fail_home(why, q_start)
        try:
            self.relay_setup()
            self.relay_clear()
            # ---- part 1: D11 -> desk
            if a.layout_poses:
                d11 = self.relay_layout()['d11']
            else:
                try:
                    guess = self.relay_d11_file()
                    self.say(f'camera over D11 (layout pose {a.relay_d11}: '
                             f'{guess["h"][0] * 1000:.0f}, {guess["h"][1] * 1000:.0f}, '
                             f'{guess["h"][2] * 1000:.0f} mm)')
                    self.look_last((guess['h'][0], guess['h'][1],
                                    guess['h'][2] - a.tray_height), 'shelf')
                except (OSError, KeyError, TypeError, ValueError) as e:
                    self.say(f'no D11 layout pose ({e}): measuring from here / scanning the shelf')
                try:
                    pin = self.detect_pin(region='shelf')
                except Stop as e:
                    if self._abort.is_set():
                        raise
                    back(f'tray on D11 not measured: {e}')
                try:
                    self.check_tray()
                except Stop as e:
                    back(str(e))
                d11 = dict(h=np.array(pin['p'], float), lr=float(pin['yaw']))
            try:
                A = self.relay_plan_a(d11)
                self.relay_summary(A['steps'], 'D11 -> desk')
                self.relay_verify(A['steps'], 'D11 -> desk')
            except Stop as e:
                if self._abort.is_set():
                    raise
                back(str(e))
            self.relay_start_tray(A['steps'])
            first = next(d for k, d, _ in A['steps'] if k == 'traj')[0]
            self.confirm('to the start in front of D11 (reversed arm, fingers closed)')
            if a.go:
                self.relay_go_to(first, 'in front of D11 (reversed arm, fingers closed)',
                                 grip=CLOSED)
            self.relay_run(A['steps'])
            self.say('tray on the desk' + ('' if a.go else ' (dry run)'))
            # ---- part 2: desk -> TurtleBot, from the TurtleBot measured now
            if a.layout_poses:
                bot = self.relay_layout()['bot_place']
            else:
                self.find_bot()
                bot = self.relay_bot_place()
            q_now = self.node.q() if a.go else A['q_up']
            try:
                bot['h'] = self.relay_check_box_r(bot['h'], 'TurtleBot box', place=True,
                                                   lr=bot['lr'])
                B = self.relay_plan_b(A, bot, q_now)
                self.relay_summary(B['steps'], 'desk -> TurtleBot')
                self.relay_verify(B['steps'], 'desk -> TurtleBot')
            except Stop as e:
                if self._abort.is_set():
                    raise
                back(f'{e}\n  the tray stays on the desk at ({A["spot"][0]:+.2f}, '
                     f'{A["spot"][1]:+.2f})')
            finally:
                if a.go:          # the tray on the desk stays an obstacle
                    self._scene(world=[self._tray_world(A['q_rel'])])
            self.relay_run(B['steps'])
            self.say('tray on the TurtleBot' + ('' if a.go else ' (dry run)'))
            self.relay_place_check('in the TurtleBot box', bot['h'], bot['lr'],
                                   float(bot['h'][2]) - a.tray_height, B['steps'],
                                   view=a.bot_close_view or None,
                                   extra=f'bot r {np.hypot(*bot["h"][:2]) * 1000:.0f} mm')
            self.relay_home(CLOSED)              # closed over the box (last step)
            self._remove_world('relay_tray')
        finally:
            a.table_z = keep
            self.track_mode = ''

    def _relay_desk_tray(self):
        """The tray standing on the desk: --layout-poses (simulation) -> --relay-desk or None
        (the planner picks a desk spot); else measured with the camera from here (d) and
        checked to stand on the desk."""
        a = self.args
        if a.layout_poses:
            return self.relay_desk_cands()[0] if a.relay_desk else None
        pin = self._detect_at_desk_spots()
        self.check_tray()
        d = dict(h=np.array(pin['p'], float), lr=float(pin['yaw']))
        dz = float(d['h'][2]) - a.tray_height - self._desk_geo[3]
        if abs(dz) > 0.02:
            raise Stop(f'the tray seen stands {dz * 1000:+.0f} mm off the desk top: not on '
                       'the desk (or wrong --tray-height / --tray-type)')
        return d

    def _detect_at_desk_spots(self):
        """e: the tray looked for ONLY at the desk spot(s) (--relay-desk, else --relay-spots,
        else the first automatic one): camera over each, measured there (small scan around
        it, --scan-step); none there -> Stop (the caller goes home with an ERROR)."""
        a = self.args
        if a.relay_desk:
            spots = [tuple(a.relay_desk[:2])]
        elif a.relay_spots:
            spots = [tuple(s) for s in a.relay_spots]
        else:
            spots = [tuple(d['h'][:2]) for d in self.relay_desk_cands()[:1]]
        z = self._desk_geo[3]
        keep = (a.search_x, a.search_y, a.use_last)
        why = []
        try:
            for x, y in spots:
                a.search_x, a.search_y, a.use_last, a.table_z = x, y, False, z
                self.say(f'camera over the desk spot ({x * 1000:.0f}, {y * 1000:.0f}) mm')
                if a.go:
                    self.look_over(x, y, z, 'desk spot')
                try:
                    return self.detect_pin(region='desk')   # no such region: no wide scan
                except Stop as e:
                    if self._abort.is_set():
                        raise
                    why.append(f'({x * 1000:.0f}, {y * 1000:.0f}) mm: {str(e).splitlines()[0]}')
        finally:
            a.search_x, a.search_y, a.use_last = keep
        raise Stop('no tray at the desk spot(s) (--relay-spots / --relay-desk):\n  '
                   + '\n  '.join(why))

    def _bot_spots(self):
        """e / r: --relay-spots-bot as the desk spots, unless --relay-spots / --relay-desk is
        given. -> the --relay-spots value to restore."""
        a = self.args
        keep = a.relay_spots
        if a.relay_spots is None and a.relay_desk is None and a.relay_spots_bot:
            a.relay_spots = list(a.relay_spots_bot)
        return keep

    @staticmethod
    def _held_travel(steps):
        """Joint travel while the tray is held: (J1, J6) [deg] summed over the held paths."""
        held, prev, j1, j6 = False, None, 0.0, 0.0
        for kind, data, _ in steps:
            if kind == 'grasp':
                held = True
            elif kind == 'release':
                held = False
            elif kind == 'traj' and data:
                pts = ([prev] if prev is not None else []) + list(data)
                if held and len(pts) > 1:
                    d = np.abs(np.diff(np.degrees(np.asarray(pts, float)), axis=0))
                    j1, j6 = j1 + float(d[:, 0].sum()), j6 + float(d[:, 5].sum())
                prev = data[-1]
        return j1, j6

    def relay_place_check(self, where, xy, lr, surface_z, steps, view=None, extra=''):
        """After a put-down (real robot, --relay-place-check): camera over the tray, measured
        again, the error vs the plan (handle midpoint xy, L -> R lr mod 180) said and logged
        (--relay-check-log) with the held joint travel. Not measured -> said, no error (the
        tray is already put down)."""
        a = self.args
        if not (a.go and a.relay_place_check) or a.layout_poses or self.conn in ('fake', None):
            return
        keep = (a.search_x, a.search_y, a.use_last, a.table_z)
        pin, why = None, ''
        try:
            a.search_x, a.search_y, a.use_last, a.table_z = xy[0], xy[1], False, surface_z
            self.say(f'place check: camera over the tray {where}')
            self.look_over(float(xy[0]), float(xy[1]), surface_z, f'place check ({where})',
                           view=view)
            pin = self.detect_pin(region='desk')     # no such region: no wide scan
        except Stop as e:
            if self._abort.is_set():
                raise
            why = str(e).splitlines()[0]
        finally:
            a.search_x, a.search_y, a.use_last, a.table_z = keep
        j1, j6 = self._held_travel(steps)
        if pin is None:
            self.say(f'place check {where}: tray not measured ({why})')
            return
        m = np.asarray(pin['p'][:2], float)
        d = m - np.asarray(xy[:2], float)
        dist = float(np.linalg.norm(d))
        dyaw = math.degrees(math.remainder(float(pin['yaw']) - lr, math.pi))   # -90..90
        wp, wy = a.relay_check_warn[0], a.relay_check_warn[1]
        bad = dist > wp or abs(dyaw) > wy
        self.say(f'{"WARNING: " if bad else ""}place check {where}: {dist * 1000:.1f} mm '
                 f'(dx {d[0] * 1000:+.1f}, dy {d[1] * 1000:+.1f}), yaw {dyaw:+.1f} deg off the '
                 f'plan (held travel J1 {j1:.0f} deg, J6 {j6:.0f} deg)'
                 + (f' > {wp * 1000:.0f} mm / {wy:.0f} deg: slipped in the fingers or pushed?'
                    if bad else ''))
        path = os.path.expanduser(a.relay_check_log or '')
        if not path:
            return
        try:
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['time', 'mode', 'where', 'plan_x_mm', 'plan_y_mm', 'plan_lr_deg',
                                'meas_x_mm', 'meas_y_mm', 'meas_lr_deg', 'dx_mm', 'dy_mm',
                                'dist_mm', 'dyaw_deg', 'tilt_deg', 'held_j1_deg',
                                'held_j6_deg', 'extra'])
                w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), getattr(self, 'track_mode', ''),
                            where, f'{xy[0] * 1000:.1f}', f'{xy[1] * 1000:.1f}',
                            f'{math.degrees(lr):.1f}', f'{m[0] * 1000:.1f}', f'{m[1] * 1000:.1f}',
                            f'{math.degrees(float(pin["yaw"])):.1f}', f'{d[0] * 1000:.1f}',
                            f'{d[1] * 1000:.1f}', f'{dist * 1000:.1f}', f'{dyaw:.1f}',
                            f'{float(pin.get("tilt", 0.0)):.1f}', f'{j1:.0f}', f'{j6:.0f}', extra])
        except OSError as e:
            self.say(f'place check: log not written ({e})')

    def _desk_lr(self, A):
        """Planned tray L -> R on the desk from leg A (tool yaw yc - --grip-yaw-offset)."""
        return float(A['yc']) - math.radians(self.args.grip_yaw_offset)

    def act_desk_to_bot(self):
        """e: desk -> TurtleBot (the second part of n alone): the tray on the desk measured
        (camera, from here), the TurtleBot marker measured, desk -> TurtleBot planned (tray
        turned far from the arm base when needed) and run. No plan: nothing moves, error."""
        self.track_mode = 'e'
        a = self.args
        if self.holding:
            raise Stop('holding the tray: put it down first')
        keep = a.table_z
        q_start = self.node.q()
        spots_keep = self._bot_spots()
        try:
            self.relay_setup()
            self.relay_clear()
            self._desk_geo = self._desk()
            try:
                desk = self._relay_desk_tray()
                if a.layout_poses:
                    bot = self.relay_layout()['bot_place']
                else:
                    self.find_bot()
                    bot = self.relay_bot_place()
                bot['h'] = self.relay_check_box_r(bot['h'], 'TurtleBot box', place=True,
                                                   lr=bot['lr'])
                q_from = self.node.q()
                if q_from is None:
                    raise Stop('no joint_states')
                A, B = self.relay_plan_desk(bot, q_from, desk)
                steps = B['steps']
                self.relay_summary(steps, 'desk -> TurtleBot')
                self.relay_verify(steps, 'desk -> TurtleBot')
            except Stop as e:
                self.relay_fail_home(str(e), q_start)
            self.relay_start_tray(steps)
            self.relay_run(steps)
            self.say('tray on the TurtleBot' + ('' if a.go else ' (dry run)'))
            self.relay_place_check('in the TurtleBot box', bot['h'], bot['lr'],
                                   float(bot['h'][2]) - a.tray_height, steps,
                                   view=a.bot_close_view or None,
                                   extra=f'bot r {np.hypot(*bot["h"][:2]) * 1000:.0f} mm')
            self.relay_home(CLOSED)              # closed over the box (last step)
            self._remove_world('relay_tray')
        finally:
            a.table_z = keep
            a.relay_spots = spots_keep
            self.track_mode = ''

    def act_bot_to_desk(self):
        """r: TurtleBot -> desk: the tray in the box measured (camera), put on the desk at
        --relay-desk (else the planner picks a desk spot: --relay-spot-r, at most
        --relay-max-spots tried); planned as desk -> TurtleBot and run backwards."""
        self.track_mode = 'r'
        a = self.args
        if self.holding:
            raise Stop('holding the tray: put it down first')
        keep = a.table_z
        q_start = self.node.q()
        spots_keep = self._bot_spots()
        try:
            self.relay_setup()
            self.relay_clear()
            self._desk_geo = self._desk()
            try:
                if a.layout_poses:
                    bot = self.relay_layout()['bot_tray']
                else:
                    pin = self.measure_tray_on_bot()
                    self.check_tray()
                    bot = dict(h=np.array(pin['p'], float), lr=float(pin['yaw']))
                self.relay_check_box_r(bot['h'], 'tray in the TurtleBot box')
                q_from = self.node.q()
                if q_from is None:
                    raise Stop('no joint_states')
                A, B = self.relay_plan_desk(bot, q_from)
                steps = self.relay_reverse(B['steps'])
                if steps[-1][0] == 'fingers':     # back at the start closed: no opening there
                    steps = steps[:-1]
                # no arm switch back to where r started (the camera view over the TurtleBot):
                # from over the tray on the desk straight home (relay_home), fingers closed
                if steps[-1][0] == 'traj' and steps[-1][2][0].startswith('arm switch back'):
                    steps = steps[:-1]
                self.relay_summary(steps, 'TurtleBot -> desk')
                self.relay_verify(steps, 'TurtleBot -> desk')
            except Stop as e:
                self.relay_fail_home(str(e), q_start)
            self.relay_start_tray(steps)
            first = next(d for k, d, _ in steps if k == 'traj')[0]
            self.confirm('to the start over the TurtleBot box (fingers closed)')
            if a.go:
                self.relay_go_to(first, 'over the TurtleBot box (fingers closed)', grip=CLOSED)
            self.relay_run(steps)
            s = A['spot']
            self.say(f'tray on the desk at ({s[0] * 1000:.0f}, {s[1] * 1000:.0f}) mm'
                     + ('' if a.go else ' (dry run)'))
            self.relay_place_check('on the desk', s, self._desk_lr(A), self._desk_geo[3],
                                   steps, extra=f'from box r {np.hypot(*bot["h"][:2]) * 1000:.0f} mm')
            self.relay_home(CLOSED)
            self._remove_world('relay_tray')
        finally:
            a.table_z = keep
            a.relay_spots = spots_keep
            self.track_mode = ''

    def act_bot_to_d11(self):
        """j: TurtleBot -> shelf D11 through a desk spot (relay.py): the tray on the
        TurtleBot is measured (camera), D11 must be empty: the tray goes to the layout pose
        --relay-d11; planned as D11 -> TurtleBot and run backwards."""
        self.track_mode = 'j'
        a = self.args
        if self.holding:
            raise Stop('holding the tray: put it down first')
        keep = a.table_z
        q_start = self.node.q()
        try:
            self.relay_setup()
            self.relay_clear()
            try:
                if a.layout_poses:
                    lp = self.relay_layout()
                    bot, d11 = lp['bot_tray'], lp['d11']
                else:
                    pin = self.measure_tray_on_bot()
                    self.check_tray()
                    bot = dict(h=np.array(pin['p'], float), lr=float(pin['yaw']))
                    d11 = self.relay_d11_file()
                self.say(f'D11 (must be empty): handle to ({d11["h"][0] * 1000:.0f}, '
                         f'{d11["h"][1] * 1000:.0f}, {d11["h"][2] * 1000:.0f}) mm ({a.relay_d11})')
                self.relay_check_box_r(bot['h'], 'tray in the TurtleBot box')
                A, B = self.relay_plan(d11, bot)
                steps = self.relay_reverse(A['steps'] + B['steps'])
                self.relay_summary(steps, 'TurtleBot -> D11')
                self.relay_verify(steps, 'TurtleBot -> D11')
            except Stop as e:
                self.relay_fail_home(str(e), q_start)
            self.relay_start_tray(steps)
            first = next(d for k, d, _ in steps if k == 'traj')[0]
            self.confirm('to the start over the TurtleBot box (fingers closed)')
            if a.go:
                self.relay_go_to(first, 'over the TurtleBot box (fingers closed)', grip=CLOSED)
            self.relay_run(steps)
            self.say('tray on D11' + ('' if a.go else ' (dry run)'))
            self.relay_home(CLOSED)              # the way out of the shelf ends closed
            self._remove_world('relay_tray')
        finally:
            a.table_z = keep
            self.track_mode = ''
