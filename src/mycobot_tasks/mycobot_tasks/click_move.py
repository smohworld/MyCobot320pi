"""Move the finger tip above a point clicked in the camera window (no cube, no pick/place).

Needs move_group, the driver and the camera running (e.g. handeye.launch.py) and the arm in a
pose where the camera sees the desk (e.g. goto_lookdown, or --search).

  ros2 run mycobot_tasks click_move                    # window, plans only (dry run)
  ros2 run mycobot_tasks click_move --go --real        # window, moves the REAL robot
  ros2 run mycobot_tasks click_move --go --real --height 0.07 --refine

The clicked pixel is the camera ray hitting the desk plane (--table-z); the finger tip goes
--height above that point, pointing down, keeping its current yaw.
--refine: after the move, read the finger tip (measured joints) and re-command the remaining
error, up to --refine-iters times (removes servo sag / stopping short, not hand-eye error).

Keys: + / - raise / lower the clicked surface by 5 mm (click the TOP of an object: set
its height), 0 desk
      h home (the pose at start)  u straight up  j straight down (--down-step, stops at
--down-min-z over the desk)  x / SPACE stop the motion  q / ESC quit
"""
import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor

from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.goto_lookdown import driver_connection
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import (CAM, DOWN, TCP, Cube, Stop, _text, project,
                                          ray_plane)


class ClickMove:
    def __init__(self, node, args):
        self.node, self.args = node, args
        self.K, self.D, _ = load_intrinsics(args.camera_calibration_file)
        self.conn = driver_connection(node)
        if args.go and self.conn != 'fake' and not args.real:
            raise SystemExit(f'driver connection is {self.conn!r}, not fake: add --real '
                             'to move it')
        self.mv = MoveItClient(node)
        self.status = 'starting ...'
        self.target = None                # last goal TCP point (drawn in the window)
        self.plane_z = args.table_z + args.surface   # height of the clicked surface [m]
        self.busy = False
        self.jobs = queue.Queue()
        self._abort = threading.Event()
        t0 = time.monotonic()
        while self.node.q() is None or self.node.T('base', CAM) is None:
            if time.monotonic() - t0 > 5.0:
                raise SystemExit(f'no joint_states / TF base -> {CAM}')
            time.sleep(0.1)
        self.home = self.node.q()
        if not self.mv.wait_ready(15.0):
            raise SystemExit('move_group not available (start handeye.launch.py first)')
        self.set_table(margin=True)
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------- job queue
    def submit(self, name, fn, *a):
        if self.busy:
            self.say(f'busy - {name} ignored (x / SPACE stops the motion)')
            return False
        self.busy = True
        self._abort.clear()
        self.jobs.put((name, fn, a))
        return True

    def _worker(self):
        while True:
            name, fn, a = self.jobs.get()
            try:
                fn(*a)
            except Stop as e:
                self.say(f'{name}: {e}')
            except Exception as e:  # noqa: BLE001
                self.say(f'{name}: ERROR {e!r}')
            finally:
                self.busy = False
                self.jobs.task_done()

    def stop(self):
        self._abort.set()
        self.mv.cancel()
        self.say('STOP requested')

    def say(self, text):
        self.status = text
        print(text, flush=True)

    def _check(self):
        if self._abort.is_set():
            raise Stop('stopped')

    # ------------------------------------------------------------ helpers
    def set_table(self, margin=True):
        top = self.args.table_z + (self.args.table_margin if margin else -0.002)
        self.mv.add_box('calibration_table', [0.0, 0.0, top - 0.025], [1.5, 1.5, 0.05])

    def tcp_yaw(self):
        T = self.node.T('base', TCP)
        if T is None:
            return 0.0
        R = T[:3, :3] @ DOWN
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def ik_down(self, p, yaw, seed):
        """TCP at p pointing down, yaw + k*90 deg (least motion from seed)."""
        best = None
        for k in range(4):
            q = self.mv.compute_ik(make_T(rot_z(yaw + k * np.pi / 2) @ DOWN, p), TCP,
                                   ARM_JOINTS, seed, timeout=0.1)
            if q is not None:
                cost = float(np.max(np.abs(np.asarray(q) - seed)))
                if best is None or cost < best[0]:
                    best = (cost, np.asarray(q))
        return None if best is None else best[1]

    def goto(self, name, p, yaw, slow=False):
        """Plan the tip to p (pointing down) and move there."""
        p = np.asarray(p, float)
        seed = self.node.q()
        q = self.ik_down(p, yaw, seed)
        if q is None:
            raise Stop(f'no IK for {name} at ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, '
                       f'{p[2] * 1000:.0f}) mm (out of reach / J6 window / collision)')
        print(f'  {name:12s} J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
        self._check()
        if not self.args.go:
            self.say(f'(dry run) {name}')
            return
        self.say(f'moving: {name}')
        vel = self.args.slow_velocity if slow else self.args.velocity
        ok, msg = self.mv.move_joints(ARM_JOINTS, q, vel, self.args.accel)
        self._check()
        if not ok:
            q_now = self.node.q()
            err = np.degrees(np.max(np.abs(q_now - q))) if q_now is not None else np.inf
            if err > 3.0:
                raise Stop(f'{name}: motion failed ({msg}), joints {err:.1f} deg off')
        time.sleep(0.1)

    def tip_error(self, goal):
        T = self.node.T('base', TCP)
        return None if T is None else goal - T[:3, 3]

    # ------------------------------------------------------------ actions
    def act_move_to(self, p_desk):
        a = self.args
        goal = np.array([p_desk[0] + a.offset_x, p_desk[1] + a.offset_y,
                         self.plane_z + a.height])
        yaw = self.tcp_yaw()
        self.target = goal
        self.goto('click point', goal, yaw)
        if a.refine and a.go:
            cmd = goal.copy()
            for i in range(a.refine_iters):
                time.sleep(a.refine_settle)
                e = self.tip_error(goal)
                if e is None:
                    break
                self.say(f'align {i + 1}/{a.refine_iters}: tip vs goal dx {e[0] * 1000:+.1f} '
                         f'dy {e[1] * 1000:+.1f} dz {e[2] * 1000:+.1f} mm')
                if np.linalg.norm(e) < a.refine_min:
                    break
                if np.linalg.norm(e) > a.refine_max:
                    raise Stop('tip error > --refine-max: not correcting')
                cmd = cmd + e
                self.goto('correction', cmd, yaw, slow=True)
        e = self.tip_error(goal)
        if e is not None and a.go:
            self.say(f'finger tip vs goal (model, measured joints): xy '
                     f'{np.linalg.norm(e[:2]) * 1000:.1f} mm, z {-e[2] * 1000:+.1f} mm')
        self.say(f'above ({goal[0] * 1000:.0f}, {goal[1] * 1000:.0f}) mm, finger tip '
                 f'{a.height * 1000:.0f} mm over the clicked surface '
                 f'({(self.plane_z - a.table_z) * 1000:.0f} mm over the desk)')

    def _step_z(self, dz, name):
        a = self.args
        T = self.node.T('base', TCP)
        if T is None:
            raise Stop(f'no TF base -> {TCP}')
        p = T[:3, 3].copy()
        p[2] = max(a.table_z + a.down_min_z, p[2] + dz)
        self.target = p
        self.set_table(margin=False)
        try:
            self.goto(name, p, self.tcp_yaw(), slow=True)
        finally:
            self.set_table(margin=True)
        self.say(f'{name}: finger tip {(p[2] - a.table_z) * 1000:.0f} mm over the desk')

    def act_up(self):
        self._step_z(+self.args.down_step, 'up')

    def act_down(self):
        a = self.args
        T = self.node.T('base', TCP)
        if T is not None and T[2, 3] <= a.table_z + a.down_min_z + 1e-4:
            raise Stop(f'already at the lowest allowed height ({a.down_min_z * 1000:.0f} mm '
                       'over the desk, --down-min-z)')
        self._step_z(-a.down_step, 'down')

    def act_home(self):
        self.target = None
        self._check()
        if not self.args.go:
            self.say('(dry run) home')
            return
        self.mv.move_joints(ARM_JOINTS, self.home, self.args.velocity, self.args.accel)
        self.say('home')

    # ------------------------------------------------------------- clicks
    def click(self, u, v):
        T_bc = self.node.T('base', CAM)
        if T_bc is None:
            self.say(f'no TF base -> {CAM}')
            return
        # the ray must go down and hit the desk in front of the camera, not far away: a
        # camera close to the desk looking sideways gives rays that miss the desk or hit it
        # very far away (the error then explodes)
        n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.D).reshape(2)
        d = T_bc[:3, :3] @ np.array([n[0], n[1], 1.0])
        z_cam = T_bc[2, 3] - self.plane_z
        if d[2] > -0.05 or z_cam < 0.03:
            self.say('click ignored: that ray does not hit the desk in front of the camera '
                     f'(camera {z_cam * 1000:.0f} mm over the desk, ray dz {d[2]:+.2f})')
            return
        p = ray_plane([u, v], self.K, self.D, T_bc, self.plane_z)[0]
        reach = float(np.hypot(p[0] + self.args.offset_x, p[1] + self.args.offset_y))
        if reach > self.args.max_reach:
            self.say(f'click ignored: {reach * 1000:.0f} mm from the base axis > --max-reach '
                     f'{self.args.max_reach * 1000:.0f} mm')
            return
        print(f'click ({u}, {v}) px -> base ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, '
              f'{p[2] * 1000:.0f}) mm')
        self.submit('move', self.act_move_to, p)


# ------------------------------------------------------------------ window
def view(app):
    node, args = app.node, app.args
    win = 'click_move'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(ev, u, v, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            app.click(u, v)

    cv2.setMouseCallback(win, on_mouse)
    keys = {ord('h'): ('home', app.act_home), ord('u'): ('up', app.act_up),
            ord('j'): ('down', app.act_down)}
    after = 0.0
    while True:
        img = node.fresh(after, timeout=0.1)
        if img is not None:
            after = time.monotonic()
            vis = img.copy()
            T_bc = node.T('base', CAM)
            if T_bc is not None and app.target is not None:
                floor = app.target.copy()
                floor[:2] -= [args.offset_x, args.offset_y]   # draw the clicked point
                floor[2] = app.plane_z        # the x is where the click is
                uv = project(floor, T_bc, app.K, app.D)
                if uv:
                    cv2.drawMarker(vis, uv, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            _text(vis, app.status, (10, 25), 0.55)
            _text(vis, f"{'' if args.go else 'DRY RUN  '}{'BUSY  ' if app.busy else ''}"
                       f'click = move above the point  (surface +{(app.plane_z - args.table_z) * 1000:.0f} mm)',
                  (10, 50), 0.5)
            _text(vis, 'h home  u up  j down  +/- surface 5 mm  0 desk  x stop  q quit', (10, vis.shape[0] - 12), 0.45)
            cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            break
        if k in (ord('x'), ord(' ')):
            app.stop()
        elif k in (ord('+'), ord('=')):
            app.plane_z += 0.005
            app.say(f'clicked surface: {(app.plane_z - args.table_z) * 1000:.0f} mm over the desk')
        elif k in (ord('-'), ord('_')):
            app.plane_z = max(args.table_z, app.plane_z - 0.005)
            app.say(f'clicked surface: {(app.plane_z - args.table_z) * 1000:.0f} mm over the desk')
        elif k == ord('0'):
            app.plane_z = args.table_z
            app.say('clicked surface: the desk')
        elif k in keys:
            app.submit(*keys[k])
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyWindow(win)


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--height', type=float, default=0.05,
                   help='finger tip above the clicked surface (desk + --surface) [m]')
    a.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    a.add_argument('--table-margin', type=float, default=0.012,
                   help='desk collision box raised by this [m]')
    a.add_argument('--offset-x', type=float, default=0.0,
                   help='measured constant error: add this to every target x in base [m] '
                        '(robot goes 10 mm too far in +x -> -0.010)')
    a.add_argument('--offset-y', type=float, default=0.0, help='same for y [m]')
    a.add_argument('--surface', type=float, default=0.0,
                   help='height of the clicked surface over the desk [m] (an object top); '
                        'keys + / - change it by 5 mm, 0 resets')
    a.add_argument('--max-reach', type=float, default=0.38,
                   help='ignore clicks whose point is farther than this from the base axis [m]')
    a.add_argument('--down-step', type=float, default=0.01, help='keys u/j: step [m]')
    a.add_argument('--down-min-z', type=float, default=0.01,
                   help='key j: lowest finger tip height over the desk [m]')
    a.add_argument('--refine', action='store_true',
                   help='re-command the remaining finger tip error after the move')
    a.add_argument('--refine-iters', type=int, default=3)
    a.add_argument('--refine-settle', type=float, default=0.5, help='wait before reading [s]')
    a.add_argument('--refine-min', type=float, default=0.001, help='good enough [m]')
    a.add_argument('--refine-max', type=float, default=0.02, help='give up above this [m]')
    a.add_argument('--velocity', type=float, default=0.6, help='velocity scaling')
    a.add_argument('--slow-velocity', type=float, default=0.25,
                   help='velocity scaling for corrections and u/j')
    a.add_argument('--accel', type=float, default=0.5, help='acceleration scaling')
    a.add_argument('--image-topic', default='/camera/image_raw')
    a.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    a.add_argument('--go', action='store_true', help='really move (else plan only)')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    args = a.parse_args()

    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise SystemExit('no display: the camera window is needed to click')
    rclpy.init()
    node = Cube(args.image_topic)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        app = ClickMove(node, args)
        app.say('ready' + ('' if args.go else ' (DRY RUN: add --go to move)'))
        view(app)
        app.jobs.join()
    except KeyboardInterrupt:
        if 'app' in locals():
            app.stop()
    finally:
        cv2.destroyAllWindows()
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
