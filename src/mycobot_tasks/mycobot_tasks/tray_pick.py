"""Pick the tray by its grip pin, from above, keeping the tray level.

Needs (like red_cube_hover) move_group, the driver and the camera running (handeye.launch.py),
the arm in a look-down pose (goto_lookdown / --search), and `tray_detect` running in
another terminal (it publishes /tray/pin_pose: centre of the graspable part of the pin and
the tray yaw, tray +x pointing away from the pin).

  ros2 run mycobot_tasks tray_pick                       # window, plans only (dry run)
  ros2 run mycobot_tasks tray_pick --go --real           # window, moves the REAL robot
  ros2 run mycobot_tasks tray_pick --go --real --search --do pick

Window: left click on the desk = move the finger tips --click-height above that point
        p pick   v move above the pin (approach height)   h home   s search pose (new home)
        d detect again   o / c open / close gripper   u straight up   x / SPACE stop   q quit

Pick: above the pin (finger tips --approach = 70 mm above the desk) -> open (--open-position,
1.0 = fully open, 10 cm) -> straight down until the finger tips are --grasp-z above the desk
(17 mm: the fingers cover the graspable part, 17..45 mm) -> close (the force gripper must
report "object grasped", else it opens again) -> lift straight up to the approach height. The
tool orientation is the same on the way up, so the tray stays level. Placing it on the
TurtleBot is not part of this node yet.

The fingers pinch the 20 mm wide pin. They close along the TCP x axis (URDF: finger joints at
gripper_base x = -44.5 / +44.2 mm), so the TCP yaw is the tray yaw + --grip-yaw-offset
(default 0; it was 90, which closes across the pin along the tray y axis). Check it in a dry
run (the plan prints the joints) and with the gripper open ABOVE the pin ('v') before the first grasp. The pin sticks out only 12 mm
of the tray wall: x error of the finger tips beyond ~3 mm drives them into the wall.
Without --go every action is only planned (IK) and printed.
"""
import argparse
import collections
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import (App, CAM, Cube, DOWN, Stop, TCP, _text, project,
                                                ray_plane)


class PinNode(Cube):
    """Cube node (images, TF, joint states, gripper client) + the /tray/pin_pose subscriber."""

    def __init__(self, topic):
        super().__init__(topic)
        self.pins = collections.deque(maxlen=100)       # (t, x, y, z, yaw)
        self.create_subscription(PoseStamped, '/tray/pin_pose', self._on_pin, 10,
                                 callback_group=ReentrantCallbackGroup())

    def _on_pin(self, m):
        q, p = m.pose.orientation, m.pose.position
        self.pins.append((time.monotonic(), p.x, p.y, p.z, 2.0 * math.atan2(q.z, q.w)))


class TrayApp(App):
    def __init__(self, node, args):
        self.pin = None                   # dict(p=xyz of the pin, yaw) of the last detection
        super().__init__(node, args)

    # ----------------------------------------------------------- detection
    def detect_cube(self):                # name kept: inherited actions (home, search) call it
        return self.detect_pin()

    def detect_pin(self):
        """Median of /tray/pin_pose received while the arm stands still -> self.pin."""
        a = self.args
        q0, t0 = self.node.q(), time.monotonic()
        while time.monotonic() - t0 < a.pin_timeout:
            if sum(1 for p in list(self.node.pins) if p[0] > t0) >= a.frames:
                break
            time.sleep(0.05)
        got = [p for p in list(self.node.pins) if p[0] > t0]
        q1 = self.node.q()
        if q0 is None or q1 is None or np.max(np.abs(q1 - q0)) > np.radians(0.5):
            raise Stop('the arm moved during detection: try again when it stands still')
        if len(got) < a.frames:
            self.pin = None
            raise Stop(f'no /tray/pin_pose ({len(got)} of {a.frames} messages in '
                       f'{a.pin_timeout:.0f} s): is tray_detect running and seeing the tray + pin?')
        arr = np.array([g[1:] for g in got])
        xy = np.median(arr[:, :2], axis=0)
        spread = float(np.max(np.linalg.norm(arr[:, :2] - xy, axis=1)))
        yaw = math.atan2(np.median(np.sin(arr[:, 3])), np.median(np.cos(arr[:, 3])))
        if spread > a.max_spread:
            self.pin = None
            raise Stop(f'pin pose not stable ({spread * 1000:.1f} mm spread > --max-spread '
                       f'{a.max_spread * 1000:.0f} mm): the tray or the arm is moving?')
        self.pin = dict(p=np.array([xy[0], xy[1], float(np.median(arr[:, 2]))]), yaw=yaw)
        self.say(f'pin ({xy[0] * 1000:.1f}, {xy[1] * 1000:.1f}) mm, tray yaw '
                 f'{math.degrees(yaw):.1f} deg, spread {spread * 1000:.1f} mm')
        return self.pin

    # ---------------------------------------------------------------- IK
    def ik_down(self, p, yaw, seed):
        """TCP at p pointing down with the yaw (or yaw + 180 deg: same pinch, least motion).
        Not yaw + 90: the fingers must keep the same pinch direction."""
        best = None
        for k in (0, 1):
            y = yaw + k * math.pi
            q = self.mv.compute_ik(make_T(rot_z(y) @ DOWN, p), TCP, ARM_JOINTS, seed,
                                   timeout=0.1)
            if q is not None:
                cost = float(np.max(np.abs(np.asarray(q) - seed)))
                if best is None or cost < best[0]:
                    best = (cost, np.asarray(q), y)
        return None if best is None else (best[1], best[2])

    # ----------------------------------------------------------- actions
    def grip_points(self):
        a = self.args
        if a.grasp_z <= a.table_margin:
            raise Stop(f'--grasp-z {a.grasp_z} must be above --table-margin {a.table_margin}')
        pin = self.pin or self.detect_pin()
        x, y = pin['p'][:2]
        yaw = pin['yaw'] + math.radians(a.grip_yaw_offset)
        return (np.array([x, y, a.table_z + a.approach]),
                np.array([x, y, a.table_z + a.grasp_z]), yaw)

    def act_hover(self):
        above, _, yaw = self.grip_points()
        self.target = above
        (n, q, _), = self.plan([('above pin', above, yaw)])
        self.move(n, q)
        self.tip_report(above)
        self.say(f'above the pin: finger tips {self.args.approach * 1000:.0f} mm over the desk')

    def act_pick(self):
        a = self.args
        if self.holding:
            raise Stop('already holding the tray')
        above, grasp, yaw = self.grip_points()
        self.target = grasp
        (n1, q_above, _), (n2, q_grasp, _) = self.plan(
            [('pick above', above, yaw), ('grasp', grasp, yaw)])
        opened = self.gripper_async(True)           # opens while the arm travels
        self.move(n1, q_above)
        opened()
        self.move(n2, q_grasp, slow=True, velocity=a.grasp_velocity)
        self.tip_report(grasp)
        r = self.gripper(False)
        missed = r is not None and not r.stalled
        if missed:
            self.gripper(True)
        else:
            self.holding = True
        self.set_table(margin=False)
        try:
            self.move('lift', q_above, slow=True)   # same orientation: the tray stays level
        finally:
            self.set_table(margin=True)
        if missed:
            raise Stop('gripper closed without the pin (missed): it stays open; '
                       'd = detect again')
        self.pin = None
        self.say('tray picked' + ('' if a.go else ' (dry run)') +
                 f': lifted level to {a.approach * 1000:.0f} mm (finger tips)')


    # ------------------------------------------------------------ clicks
    def click(self, u, v, _on_cube=False):
        """Left click on the desk: move the finger tips --click-height above that point
        (keeps the current tool yaw). The tray is never placed by a click."""
        if self.holding:
            self.say('holding the tray: a click does not place it (not implemented yet)')
            return
        T_bc = self.node.T('base', CAM)
        if T_bc is None:
            self.say(f'no TF base -> {CAM}')
            return
        p = ray_plane([u, v], self.K, self.D, T_bc, self.args.table_z)[0]
        axis = T_bc[:3, 2]                         # optical axis in base
        tilt = math.degrees(math.acos(min(1.0, abs(axis[2]))))
        print(f'click px ({u}, {v}) -> desk point ({p[0] * 1000:.1f}, {p[1] * 1000:.1f}) mm '
              f'at z {self.args.table_z * 1000:.0f} mm | camera at ({T_bc[0, 3] * 1000:.0f}, '
              f'{T_bc[1, 3] * 1000:.0f}, {T_bc[2, 3] * 1000:.0f}) mm, optical axis '
              f'{tilt:.1f} deg from vertical', flush=True)
        self.submit('move', self.act_move_to, p)


# ------------------------------------------------------------------ window
def view(app):
    node, args = app.node, app.args
    win = 'tray_pick'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(ev, u, v, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            app.click(u, v)

    cv2.setMouseCallback(win, on_mouse)
    keys = {ord('p'): ('pick', app.act_pick), ord('v'): ('above pin', app.act_hover),
            ord('h'): ('home', app.act_home), ord('d'): ('detect', app.act_detect),
            ord('u'): ('up', app.act_up), ord('s'): ('search pose', app.act_search),
            ord('o'): ('open', lambda: app.gripper(True)),
            ord('c'): ('close', lambda: app.gripper(False))}
    after = 0.0
    while True:
        img = node.fresh(after, timeout=0.1)
        if img is not None:
            after = time.monotonic()
            vis = img.copy()
            T_bc = node.T('base', 'camera_optical_frame')
            if T_bc is not None:
                for p, col in ((app.pin['p'] if app.pin else None, (0, 255, 255)),
                               (app.target, (255, 0, 255))):
                    if p is not None:
                        uv = project(p, T_bc, app.K, app.D)
                        if uv:
                            cv2.drawMarker(vis, uv, col, cv2.MARKER_TILTED_CROSS, 18, 2)
                if app.target is not None:
                    # X above is the finger-tip goal (in the air, so it shifts in the image);
                    # the cross below is the same x,y ON the desk, where you clicked.
                    foot = np.array([app.target[0], app.target[1], args.table_z])
                    uv = project(foot, T_bc, app.K, app.D)
                    if uv:
                        cv2.drawMarker(vis, uv, (255, 0, 255), cv2.MARKER_CROSS, 18, 1)
                        cv2.circle(vis, uv, 4, (255, 0, 255), -1)
            _text(vis, app.status, (10, 25), 0.55)
            _text(vis, f"{'' if args.go else 'DRY RUN  '}{'BUSY  ' if app.busy else ''}"
                       f"{'HOLDING' if app.holding else ''}", (10, 50), 0.5)
            _text(vis, 'click = move above desk point  p pick  v above pin  h home  s search  u up  '
                       'o/c grip  d detect  x stop  q quit', (10, vis.shape[0] - 12), 0.45)
            cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            break
        if k in (ord('x'), ord(' ')):
            app.stop()
        elif k in keys:
            app.submit(*keys[k])
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyWindow(win)


# -------------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--approach', type=float, default=0.07,
                   help='finger tips above the desk for the approach / lift [m]')
    a.add_argument('--click-height', type=float, default=0.05,
                   help='finger tips above the desk for a clicked desk point [m]')
    a.add_argument('--grasp-z', type=float, default=0.017,
                   help='finger tips above the desk when grasping [m] (> --table-margin); '
                        '17 mm: the fingers cover the graspable part 17..45 mm')
    a.add_argument('--grip-yaw-offset', type=float, default=0.0,
                   help='TCP yaw relative to the tray x axis [deg]; the fingers close along '
                        'the TCP x axis, so 0 = they close along the tray x axis, 90 = along the tray y axis '
                        '(the former default). 0 and 180 pinch the same way')
    a.add_argument('--open-position', type=float, default=1.0,
                   help='gripper open command (joint value, 1.0 = fully open = 10 cm)')
    a.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    a.add_argument('--table-margin', type=float, default=0.012,
                   help='desk collision box raised by this [m]')
    a.add_argument('--frames', type=int, default=5, help='/tray/pin_pose messages used')
    a.add_argument('--pin-timeout', type=float, default=4.0, help='wait for them [s]')
    a.add_argument('--max-spread', type=float, default=0.006,
                   help='refuse when the pin xy jumps more than this between messages [m]')
    a.add_argument('--image-topic', default='/camera/image_raw')
    a.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    a.add_argument('--velocity', type=float, default=0.6,
                   help='velocity scaling for the free moves')
    a.add_argument('--slow-velocity', type=float, default=0.25,
                   help='velocity scaling near the desk: grasp descent and lift')
    a.add_argument('--grasp-velocity', type=float, default=None,
                   help='velocity scaling of the descent only (default: --slow-velocity)')
    a.add_argument('--accel', type=float, default=0.5, help='acceleration scaling')
    a.add_argument('--search', action='store_true',
                   help='go to the search pose first (it becomes home), then --do / detect')
    a.add_argument('--search-x', type=float, default=0.25)
    a.add_argument('--search-y', type=float, default=0.0)
    a.add_argument('--search-height', type=float, default=0.28)
    a.add_argument('--search-min-height', type=float, default=0.18)
    a.add_argument('--do', choices=['', 'none', 'detect', 'hover', 'pick'], default='',
                   help='run this at start (none = nothing, e.g. for click-to-move without tray_detect)')
    a.add_argument('--go', action='store_true', help='really move (else plan only)')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    a.add_argument('--no-show', action='store_true', help='no camera window')
    a.add_argument('--holding', action='store_true',
                   help='start as holding the tray (after a restart)')
    args, _ = a.parse_known_args()             # ignore --ros-args from ros2 run
    args.return_home = False
    show = not args.no_show
    if show and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        print('no display: running without the camera window')
        show = False
    if not show and not args.do:
        args.do = 'detect'

    rclpy.init()
    node = PinNode(args.image_topic)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        app = TrayApp(node, args)
        app.say('ready' + ('' if args.go else ' (DRY RUN: add --go to move)'))
        first = {'detect': ('detect', app.act_detect), 'hover': ('above pin', app.act_hover),
                 'pick': ('pick', app.act_pick)}
        if args.holding and not args.do:
            args.do = 'none'
        if args.search:
            name, fn = first.get(args.do, ('detect', None))
            app.submit('search pose', app.act_search, fn)
        elif args.do != 'none':
            app.submit(*first.get(args.do, ('detect', app.act_detect)))
        if show:
            view(app)
            if app.busy:
                print('window closed: waiting for the running action (Ctrl+C stops)')
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
