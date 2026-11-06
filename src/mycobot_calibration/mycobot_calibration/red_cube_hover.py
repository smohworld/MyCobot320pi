"""Red cube on the desk, seen by the flange camera: hover above it, pick and place it, or
move to a point clicked in the camera window.

Needs move_group, the driver and the camera running (e.g. handeye.launch.py) and the arm
in a pose where the camera sees the desk (e.g. goto_lookdown): that start pose is "home".
--search (or key s) instead moves to the search pose first and makes it "home": the camera
--search-height above the desk point (--search-x, --search-y; default straight in front of
the base on its +x axis, the direction the arm faces with all joints 0), looking straight
down. Of the IK solutions: J1 closest to that direction first, then the least motion from the
current pose (at program start: all joints 0).

  ros2 run mycobot_calibration red_cube_hover                  # window, plans only (dry run)
  ros2 run mycobot_calibration red_cube_hover --go --real      # window, moves the REAL robot
  ros2 run mycobot_calibration red_cube_hover --go --real --do pickplace --place 0.25,0.08
  ros2 run mycobot_calibration red_cube_hover --go --real --do hover --no-show
  ros2 run mycobot_calibration red_cube_hover --go --real --search          # search pose first

Camera window (live contour / centre of the red blob, yellow = detected cube top):
  left click on the cube   pick it
  left click on the desk   holding the cube: place it there, else move above that point
  p pick   v hover above the cube   h home   s search pose (new home)   u straight up
  o / c open / close gripper   d detect again (not while holding)
  x or SPACE  stop the motion      m red mask view      q / ESC quit
Without --go every action is only planned (IK) and printed.

Position: the cube stands on the desk (--table-z), so the camera ray through the centroid
of the red blob is intersected with the plane at the cube's mid height (the silhouette
centroid is ~ the projection of the cube centre). Yaw from the blob's min-area rectangle.
A clicked desk point is the camera ray through the click hitting the desk plane.

Pick: above the cube (--approach) -> open -> straight down until the finger tips are
--grasp-z above the desk -> close (force gripper reports "object grasped") -> lift -> home.
--refine: detect the cube again from above it (close to the grasp pose, so most of the
kinematic / hand-eye error cancels) and re-plan the descent; the shift is printed.
Place: above the point -> down to --grasp-z (+ --place-drop) -> open -> lift -> home ->
detect. The servos stop up to ~3-4 deg off target before the driver corrects (~15-20 mm at
the finger tip): first runs with a fake robot or a higher --grasp-z.
"""
import argparse
import itertools
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import GripperCommand
from cv_bridge import CvBridge
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import inv, look_at, make_T, rot_z, transform_msg_to_T
from mycobot_calibration.goto_lookdown import LIMITS, SEEDS_J23, driver_connection
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_calibration.moveit_client import MoveItClient, _wait

DOWN = np.diag([1.0, -1.0, -1.0])        # TCP z axis pointing down (as in verify_touch)
CAM = 'camera_optical_frame'
TCP = 'gripper_tcp'


class Stop(Exception):
    """An action could not go on (shown in the window / printed)."""


# ------------------------------------------------------------------ vision
def red_mask(bgr, s_min, v_min, hue_width):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, s_min, v_min), (hue_width, 255, 255))
    hi = cv2.inRange(hsv, (180 - hue_width, s_min, v_min), (180, 255, 255))
    m = cv2.bitwise_or(lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def detect(bgr, args):
    """Largest red blob: (centroid_px, minAreaRect corners px (4x2), area_px, contour)."""
    m = red_mask(bgr, args.s_min, args.v_min, args.hue_width)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < args.min_area:
        return None
    mo = cv2.moments(c)
    return (np.array([mo['m10'] / mo['m00'], mo['m01'] / mo['m00']]),
            cv2.boxPoints(cv2.minAreaRect(c)), area, c)


def ray_plane(px, K, D, T_base_cam, z):
    """Base-frame point(s) where the camera ray through pixel(s) px hits the plane z."""
    n = cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    d = T_base_cam[:3, :3] @ np.column_stack([n, np.ones(len(n))]).T      # 3 x N
    o = T_base_cam[:3, 3:4]
    s = (z - o[2]) / d[2]
    return (o + d * s).T


def project(p, T_base_cam, K, D):
    pc = T_base_cam[:3, :3].T @ (np.asarray(p) - T_base_cam[:3, 3])
    if pc[2] < 0.01:
        return None
    px, _ = cv2.projectPoints(pc.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    return tuple(int(round(v)) for v in px.reshape(2))


# -------------------------------------------------------------------- node
class Cube(Node):
    def __init__(self, topic):
        super().__init__('red_cube_hover')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.img = None
        self.js = {}
        cb = ReentrantCallbackGroup()
        self.create_subscription(Image, topic, self._on_img, 2, callback_group=cb)
        self.create_subscription(JointState, 'joint_states',
                                 lambda m: self.js.update(zip(m.name, m.position)), 10,
                                 callback_group=cb)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.grip = ActionClient(self, GripperCommand, 'gripper_controller/gripper_cmd',
                                 callback_group=cb)

    def _on_img(self, msg):
        with self.lock:
            self.img = (time.monotonic(), self.bridge.imgmsg_to_cv2(msg, 'bgr8'))

    def fresh(self, after, timeout=3.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if self.img is not None and self.img[0] > after:
                    return self.img[1]
            time.sleep(0.02)
        return None

    def T(self, parent, child):
        try:
            return transform_msg_to_T(
                self.tf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def q(self):
        return np.array([self.js[n] for n in ARM_JOINTS]) if all(
            n in self.js for n in ARM_JOINTS) else None


# --------------------------------------------------------------------- app
class App:
    def __init__(self, node, args):
        self.node, self.args = node, args
        self.K, self.D, _ = load_intrinsics(args.camera_calibration_file)
        self.conn = driver_connection(node)
        if args.go and self.conn != 'fake' and not args.real:
            raise SystemExit(f'driver connection is {self.conn!r}, not fake: add --real '
                             'to move it')
        self.mv = MoveItClient(node)
        self.status = 'starting ...'
        self.cube = None                  # dict(top, centre, yaw) of the last detection
        self.target = None                # last commanded TCP point (shown in the window)
        self.holding = False
        self.grip_yaw = None              # TCP yaw used at the grasp
        self.busy = False
        self.jobs = queue.Queue()
        self._abort = threading.Event()
        t0 = time.monotonic()
        while self.node.q() is None or self.node.T('base', CAM) is None:
            if time.monotonic() - t0 > 5.0:
                raise SystemExit(f'no joint_states / TF base -> {CAM}')
            time.sleep(0.1)
        self.home = self.node.q()
        if args.holding:
            self.holding, self.grip_yaw = True, None
            self.mv.remove_object('red_cube')
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

    # ------------------------------------------------------------ helpers
    def _check(self):
        if self._abort.is_set():
            raise Stop('stopped')

    def detect_cube(self):
        """Median of several detections from the still arm -> self.cube."""
        a = self.args
        q0 = self.node.q()
        T_bc = self.node.T('base', CAM)
        cents, boxes, after = [], [], time.monotonic()
        for _ in range(a.frames * 3):
            img = self.node.fresh(after)
            if img is None:
                raise Stop(f'no image on {a.image_topic}')
            after = time.monotonic()
            det = detect(img, a)
            if det is not None:
                cents.append(det[0])
                boxes.append(det[1])
                area = det[2]
            if len(cents) >= a.frames:
                break
        q1 = self.node.q()
        if q0 is None or q1 is None or np.max(np.abs(q1 - q0)) > np.radians(0.5):
            raise Stop('the arm moved during detection: try again when it stands still')
        if len(cents) < max(1, a.frames // 2):
            self.cube = None
            self.mv.remove_object('red_cube')
            raise Stop(f'no red cube ({len(cents)} of {a.frames * 3} frames; m = mask view, '
                       'tune --s-min/--v-min/--hue-width)')
        c_px = np.median(cents, axis=0)
        z_mid, z_top = a.table_z + a.size / 2, a.table_z + a.size
        centre = ray_plane(c_px, self.K, self.D, T_bc, z_mid)[0]
        corners = ray_plane(boxes[-1], self.K, self.D, T_bc, z_top)
        e = corners[1] - corners[0]
        yaw = float(np.arctan2(e[1], e[0])) % (np.pi / 2)
        top = np.array([centre[0], centre[1], z_top])
        dist = float(np.linalg.norm(centre - T_bc[:3, 3]))
        ratio = np.sqrt(area) / (self.K[0, 0] * a.size / dist)
        self.cube = dict(top=top, centre=np.array([centre[0], centre[1], z_mid]), yaw=yaw)
        self.add_cube_box(self.cube)
        msg = (f'cube top ({top[0] * 1000:.1f}, {top[1] * 1000:.1f}, {top[2] * 1000:.1f}) mm, '
               f'yaw {np.degrees(yaw):.0f} deg, size x{ratio:.2f}')
        if not 0.7 < ratio < 1.8:
            msg += ' - WARNING: blob size does not fit the cube'
        self.say(msg)
        return self.cube

    def set_table(self, margin=True):
        """Desk collision box: raised by --table-margin (safety for the ~3-6 mm arm error),
        or at the real desk (minus 2 mm) for the moves right at the desk: a closed gripper's
        finger links hang lower in the model and sit inside the margin box, so MoveIt would
        reject the lift ("start state in collision")."""
        a = self.args
        top = a.table_z + (a.table_margin if margin else -0.002)
        self.mv.add_box('calibration_table', [0.0, 0.0, top - 0.025], [1.5, 1.5, 0.05])

    def add_cube_box(self, c):
        a = self.args
        s = a.size + 2 * a.cube_margin
        self.mv.add_box('red_cube', c['centre'] + [0, 0, a.cube_margin / 2],
                        [s, s, a.size + a.cube_margin])

    def tcp_yaw(self):
        T = self.node.T('base', TCP)
        if T is None:
            return 0.0
        R = T[:3, :3] @ DOWN                      # undo the "down" part -> rot_z(yaw)
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
                    best = (cost, np.asarray(q), yaw + k * np.pi / 2)
        return None if best is None else (best[1], best[2])

    def plan(self, steps):
        """steps: [(name, point, yaw)] -> [(name, q, yaw)], each seeded by the previous."""
        seed, out = self.node.q(), []
        for name, p, yaw in steps:
            r = self.ik_down(np.asarray(p, float), yaw, seed)
            if r is None:
                raise Stop(f'no IK for {name} at ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, '
                           f'{p[2] * 1000:.0f}) mm (out of reach / J6 window / collision)')
            out.append((name, r[0], r[1]))
            seed = r[0]
        for name, q, _ in out:
            print(f'  {name:10s} J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
        return out

    def move(self, name, q, slow=False, velocity=None):
        self._check()
        if not self.args.go:
            self.say(f'(dry run) {name}')
            return
        self.say(f'moving: {name}')
        v = velocity or (self.args.slow_velocity if slow else self.args.velocity)
        ok, msg = self.mv.move_joints(ARM_JOINTS, q, v, self.args.accel)
        self._check()
        if not ok:
            q_now = self.node.q()
            err = np.degrees(np.max(np.abs(q_now - q))) if q_now is not None else np.inf
            if err > 3.0:
                raise Stop(f'{name}: motion failed ({msg}), joints {err:.1f} deg off')
        time.sleep(0.1)

    def gripper_async(self, open_):
        """Send a gripper command without waiting (e.g. open while the arm travels).
        Returns wait() -> GripperCommand result (None in a dry run)."""
        self._check()
        pos = self.args.open_position if open_ else 0.0
        what = 'open' if open_ else 'close'
        if not self.args.go:
            self.say(f'(dry run) gripper {what}')
            return lambda: None
        if not self.node.grip.wait_for_server(timeout_sec=3.0):
            raise Stop('gripper action gripper_controller/gripper_cmd not available')
        g = GripperCommand.Goal()
        g.command.position = float(pos)
        gh = _wait(self.node.grip.send_goal_async(g), 5.0)
        if gh is None or not gh.accepted:
            raise Stop(f'gripper {what} rejected')
        fut = gh.get_result_async()

        def wait():
            res = _wait(fut, 15.0)
            if res is None:
                raise Stop(f'gripper {what}: no result')
            return res.result
        return wait

    def gripper(self, open_):
        self._check()
        pos = self.args.open_position if open_ else 0.0
        what = 'open' if open_ else 'close'
        if not self.args.go:
            self.say(f'(dry run) gripper {what}')
            return None
        self.say(f'gripper {what}')
        if not self.node.grip.wait_for_server(timeout_sec=3.0):
            raise Stop('gripper action gripper_controller/gripper_cmd not available')
        g = GripperCommand.Goal()
        g.command.position = float(pos)
        gh = _wait(self.node.grip.send_goal_async(g), 5.0)
        if gh is None or not gh.accepted:
            raise Stop(f'gripper {what} rejected')
        res = _wait(gh.get_result_async(), 15.0)
        if res is None:
            raise Stop(f'gripper {what}: no result')
        return res.result

    def tip_report(self, p_goal):
        T = self.node.T('base', TCP)
        if T is not None and self.args.go:
            d = T[:3, 3] - p_goal
            self.say(f'finger tip vs goal (model, measured joints): xy '
                     f'{np.linalg.norm(d[:2]) * 1000:.1f} mm, z {d[2] * 1000:+.1f} mm')

    # ------------------------------------------------------------ actions
    def act_detect(self):
        if self.holding:
            raise Stop('holding the cube (the camera would see it in the gripper): '
                       'place it first')
        self.detect_cube()

    def act_up(self):
        """Recovery: straight up by --approach from wherever the arm is (e.g. a failed
        lift). A cube box around the fingers (detected while held) is removed too."""
        T = self.node.T('base', TCP)
        if T is None:
            raise Stop(f'no TF base -> {TCP}')
        if self.holding or self.node.js.get('gripper_controller', 1.0) < 0.5:
            self.mv.remove_object('red_cube')
            self.cube = None
        p = T[:3, 3] + [0, 0, self.args.approach]
        self.target = p
        self.set_table(margin=False)
        try:
            r = self.ik_down(p, self.tcp_yaw(), self.node.q())
            if r is None:
                raise Stop('no IK straight above the current pose')
            self.move('up', r[0], slow=True)
        finally:
            self.set_table(margin=True)
        self.say('lifted' + (' - holding: click a desk point to place' if self.holding else ''))

    def act_home(self):
        self.target = None
        if self.holding:
            self.mv.remove_object('red_cube')
        self.move('home', self.home)
        if self.args.go and not self.holding:
            time.sleep(0.5)
            try:
                self.detect_cube()
            except Stop as e:
                self.say(str(e))
        else:
            self.say('home')

    def search_pose(self):
        """Camera looking straight down at (search_x, search_y, desk), highest first.
        Cost: J1 off the direction of the point (weighted x10, the camera sits 7.6 cm beside
        the flange so J1 is a few deg off), then the joint motion from the current pose."""
        a = self.args
        X = self.node.T('link6', CAM)
        T_l6_tcp = self.node.T('link6', TCP)
        if X is None or T_l6_tcp is None:
            raise Stop(f'no TF link6 -> {CAM} / {TCP}')
        target = np.array([a.search_x, a.search_y, a.table_z])
        min_tcp_z = a.table_z + a.table_margin + 0.02
        j1 = float(np.arctan2(a.search_y, a.search_x))
        q_now = self.node.q()
        if q_now is None:
            raise Stop('no joint_states')
        h = a.search_height
        while h >= a.search_min_height - 1e-9:
            T0 = look_at(target + [0.0, 0.0, h], target)
            best = None
            for roll in np.radians(np.arange(0, 360, 15)):
                T_ik = T0 @ make_T(rot_z(roll), np.zeros(3)) @ inv(X) @ T_l6_tcp
                if T_ik[2, 3] < min_tcp_z:
                    continue
                # several seeds: KDL IK from a single seed misses the J1 ~ front solution
                for dj1, (j2, j3) in itertools.product(
                        (0.0, -0.35, 0.35), SEEDS_J23 + [(0.3, -0.9), (-0.3, -0.6)]):
                    q = self.mv.compute_ik(T_ik, TCP, ARM_JOINTS,
                                           [j1 + dj1, j2, j3, 0.0, 0.0, 0.0], timeout=0.05)
                    if q is None:
                        continue
                    q = np.asarray(q)
                    if np.min(1 - np.abs(q) / LIMITS) <= 0.05:     # keep 5 % off the limits
                        continue
                    cost = 10.0 * (q[0] - j1) ** 2 + float(np.sum((q - q_now) ** 2))
                    if best is None or cost < best[0]:
                        best = (cost, q)
            if best is not None:
                return best[1], h
            h -= 0.02
        raise Stop(f'no search pose above ({a.search_x * 1000:.0f}, '
                   f'{a.search_y * 1000:.0f}) mm (try another --search-x / --search-height)')

    def act_search(self, then=None):
        """Move to the search pose, make it home, detect (then run `then`)."""
        if self.holding:
            raise Stop('holding the cube: place it before going to the search pose')
        self.target = None
        q, h = self.search_pose()
        self.say(f'search pose: camera {h * 100:.0f} cm above ({self.args.search_x * 1000:.0f}, '
                 f'{self.args.search_y * 1000:.0f}) mm, J1..J6 [deg] ' +
                 ' '.join(f'{v:.1f}' for v in np.degrees(q)))
        self.move('search pose', q)
        if not self.args.go:
            return
        self.home = q
        time.sleep(0.5)
        if then is not None:
            then()
            return
        try:
            self.detect_cube()
        except Stop as e:
            self.say(str(e))

    def act_hover(self):
        a = self.args
        c = self.cube or self.detect_cube()
        p = c['top'] + [0, 0, a.hover]
        self.target = p
        steps = self.plan([('approach', c['top'] + [0, 0, a.approach], c['yaw']),
                           ('hover', p, c['yaw'])])
        for name, q, _ in steps:
            self.move(name, q, slow=name == 'hover')
        self.tip_report(p)
        self.say(f'above the cube: finger tip {a.hover * 1000:.0f} mm over the top face')

    def act_move_to(self, p_desk):
        a = self.args
        p = np.array([p_desk[0], p_desk[1], a.table_z + a.click_height])
        self.target = p
        (name, q, _), = self.plan([('click point', p, self.tcp_yaw())])
        self.move(name, q)
        self.tip_report(p)
        self.say(f'above ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}) mm, finger tip '
                 f'{a.click_height * 1000:.0f} mm over the desk')

    def act_pick(self):
        a = self.args
        if self.holding:
            raise Stop('already holding the cube: click the desk to place it')
        c = self.detect_cube() if self.cube is None else self.cube
        if a.grasp_z >= a.size:
            raise Stop('--grasp-z must be below the cube height')
        above = c['top'] + [0, 0, a.approach]
        grasp = np.array([c['top'][0], c['top'][1], a.table_z + a.grasp_z])
        self.target = grasp
        # the fingers close around the cube: take it out of the planning scene first
        self.mv.remove_object('red_cube')
        try:
            steps = self.plan([('pick above', above, c['yaw']), ('grasp', grasp, c['yaw'])])
        except Stop:
            self.add_cube_box(c)              # keep it an obstacle
            raise
        (n1, q_above, yaw), (n2, q_grasp, _) = steps
        opened = self.gripper_async(True)     # opens while the arm travels
        self.move(n1, q_above)
        opened()
        q_lift = q_above
        if a.refine and a.go:
            r = self.refine_grasp(c)
            if r is not None:
                grasp, q_grasp, q_lift, yaw = r
        self.move(n2, q_grasp, slow=True, velocity=self.args.grasp_velocity)
        self.tip_report(grasp)
        r = self.gripper(False)
        missed = r is not None and not r.stalled
        if missed:
            self.gripper(True)
        else:
            self.holding, self.grip_yaw, self.cube = True, yaw, None
        self.set_table(margin=False)
        try:
            self.move('lift', q_lift, slow=True)
        finally:
            self.set_table(margin=True)
        if missed:
            raise Stop('gripper closed without the cube (missed): it stays open; '
                       'h = home, d = detect again')
        self.say('cube picked' + ('' if a.go else ' (dry run)') + ': click a desk point to '
                 'place it')
        if a.return_home:
            self.move('home', self.home)
            self.say('holding the cube at home: click a desk point to place it')

    def act_place(self, p_desk):
        a = self.args
        if not self.holding and a.go:
            raise Stop('not holding the cube (p = pick)')
        put = np.array([p_desk[0], p_desk[1], a.table_z + a.grasp_z + a.place_drop])
        above = put + [0, 0, a.approach + a.size - a.grasp_z]
        self.target = put
        yaw = self.grip_yaw if self.grip_yaw is not None else self.tcp_yaw()
        (n1, q_above, _), = self.plan([('place above', above, yaw)])
        self.move(n1, q_above)
        self.set_table(margin=False)          # closed fingers hang into the margin box
        try:
            seed = self.node.q() if self.args.go else q_above
            r = self.ik_down(put, yaw, seed)
            if r is None:
                raise Stop('no IK for put down (out of reach / J6 window)')
            q_put = r[0]
            print('  put down   J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q_put)))
            self.move('put down', q_put, slow=True)
            self.tip_report(put)
            self.gripper(True)
            self.holding, self.grip_yaw = False, None
            self.move('lift', q_above, slow=True)
        finally:
            self.set_table(margin=True)
        self.say('cube placed' + ('' if a.go else ' (dry run)'))
        if a.return_home:
            self.act_home()

    def refine_grasp(self, c0):
        """From above the cube: detect it again and re-plan grasp + lift there.
        Returns (grasp point, q_grasp, q_lift, yaw), or None to keep the first plan."""
        a = self.args
        time.sleep(a.refine_settle)
        try:
            c = self.detect_cube()
        except Stop as e:
            self.say(f'refine: {e} - keeping the first detection')
            return None
        finally:
            self.mv.remove_object('red_cube')     # detect_cube adds it back: fingers go there
        d = c['top'][:2] - c0['top'][:2]
        dyaw = (np.degrees(c['yaw'] - c0['yaw']) + 45.0) % 90.0 - 45.0
        msg = (f'refine: cube moved dx {d[0] * 1000:+.1f} dy {d[1] * 1000:+.1f} mm '
               f'(|d| {np.linalg.norm(d) * 1000:.1f} mm), yaw {dyaw:+.1f} deg')
        if np.linalg.norm(d) > a.refine_max:
            self.cube = c0
            self.add_cube_box(c0)                 # still an obstacle: nothing is grasped
            raise Stop(f'{msg} > --refine-max {a.refine_max * 1000:.0f} mm: wrong detection? '
                       'not grasping (d = detect again)')
        if np.linalg.norm(d) < a.refine_min and abs(dyaw) < 2.0:
            self.say(msg + ' - keeping the first plan')
            return None
        grasp = np.array([c['top'][0], c['top'][1], a.table_z + a.grasp_z])
        above = c['top'] + [0, 0, a.approach]
        try:
            steps = self.plan([('grasp', grasp, c['yaw']), ('lift', above, c['yaw'])])
        except Stop:
            self.add_cube_box(c)
            raise
        (_, q_grasp, yaw), (_, q_lift, _) = steps
        self.target = grasp
        self.say(msg + ' - re-planned')
        return grasp, q_grasp, q_lift, yaw

    def act_pickplace(self, p_desk):
        self.act_pick()
        self.act_place(p_desk)

    # ------------------------------------------------------------ clicks
    def click(self, u, v, on_cube):
        T_bc = self.node.T('base', CAM)
        if T_bc is None:
            self.say(f'no TF base -> {CAM}')
            return
        p = ray_plane([u, v], self.K, self.D, T_bc, self.args.table_z)[0]
        if self.holding:
            self.submit('place', self.act_place, p)
        elif on_cube:
            self.submit('pick', self.act_pick)
        else:
            self.submit('move', self.act_move_to, p)


# ------------------------------------------------------------------ window
def view(app):
    node, args = app.node, app.args
    win = 'red_cube_hover'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    state = {'det': None, 'mask': False}

    def on_mouse(ev, u, v, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        det = state['det']
        on_cube = det is not None and cv2.pointPolygonTest(det[3], (float(u), float(v)),
                                                            False) >= 0
        app.click(u, v, on_cube)

    cv2.setMouseCallback(win, on_mouse)
    keys = {ord('p'): ('pick', app.act_pick), ord('v'): ('hover', app.act_hover),
            ord('h'): ('home', app.act_home), ord('d'): ('detect', app.act_detect),
            ord('u'): ('up', app.act_up), ord('s'): ('search pose', app.act_search),
            ord('o'): ('open', lambda: app.gripper(True)),
            ord('c'): ('close', lambda: app.gripper(False))}
    after = 0.0
    while True:
        img = node.fresh(after, timeout=0.1)
        if img is not None:
            after = time.monotonic()
            vis = (cv2.cvtColor(red_mask(img, args.s_min, args.v_min, args.hue_width),
                                cv2.COLOR_GRAY2BGR) if state['mask'] else img.copy())
            det = state['det'] = detect(img, args)
            if det is not None:
                c, box, area, cnt = det
                cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
                cv2.drawContours(vis, [np.int32(box)], -1, (255, 128, 0), 1)
                p = tuple(int(round(x)) for x in c)
                cv2.drawMarker(vis, p, (0, 0, 0), cv2.MARKER_CROSS, 26, 4)
                cv2.drawMarker(vis, p, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
                x0, y0, _, _ = cv2.boundingRect(cnt)
                _text(vis, f'centre ({c[0]:.0f}, {c[1]:.0f}) px  area {area:.0f}',
                      (x0, max(15, y0 - 8)), 0.5)
            T_bc = node.T('base', CAM)
            if T_bc is not None:
                if app.cube is not None:
                    uv = project(app.cube['top'], T_bc, app.K, app.D)
                    if uv:
                        cv2.circle(vis, uv, 8, (0, 255, 255), 2)
                if app.target is not None:
                    uv = project(app.target, T_bc, app.K, app.D)
                    if uv:
                        cv2.drawMarker(vis, uv, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            mode = ('HOLDING - click desk = place' if app.holding else
                    'click cube = pick, desk = move')
            _text(vis, app.status, (10, 25), 0.55)
            _text(vis, f"{'' if args.go else 'DRY RUN  '}{'BUSY  ' if app.busy else ''}{mode}",
                  (10, 50), 0.5)
            _text(vis, 'p pick  v hover  h home  s search  u up  o/c grip  d detect  x stop  m mask  q quit',
                  (10, vis.shape[0] - 12), 0.45)
            cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            break
        if k in (ord('x'), ord(' ')):
            app.stop()
        elif k == ord('m'):
            state['mask'] = not state['mask']
        elif k in keys:
            app.submit(*keys[k])
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyWindow(win)


def _text(img, s, org, scale):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


# -------------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--size', type=float, default=0.026, help='cube edge [m]')
    a.add_argument('--hover', type=float, default=0.03,
                   help='v: finger tip above the cube top [m]')
    a.add_argument('--approach', type=float, default=0.08,
                   help='approach height above the cube top / place point [m]')
    a.add_argument('--grasp-z', type=float, default=0.015,
                   help='finger tips above the desk when grasping [m] (> --table-margin)')
    a.add_argument('--place-drop', type=float, default=0.003,
                   help='release the cube this far above the desk [m]')
    a.add_argument('--click-height', type=float, default=0.05,
                   help='finger tip above the desk for a clicked desk point [m]')
    a.add_argument('--open-position', type=float, default=1.0,
                   help='gripper open command (joint value, 1.0 = fully open)')
    a.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    a.add_argument('--table-margin', type=float, default=0.012,
                   help='desk collision box raised by this [m]')
    a.add_argument('--cube-margin', type=float, default=0.005,
                   help='cube collision box grown by this on every side [m]')
    a.add_argument('--frames', type=int, default=5, help='detections averaged')
    a.add_argument('--image-topic', default='/camera/image_raw')
    a.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    a.add_argument('--s-min', type=int, default=100, help='HSV saturation min')
    a.add_argument('--v-min', type=int, default=60, help='HSV value min')
    a.add_argument('--hue-width', type=int, default=10, help='red hue +- around 0/180')
    a.add_argument('--min-area', type=float, default=150.0, help='blob area min [px]')
    # scaling of joint_limits.yaml (0.5 rad/s = 29 deg/s, 1.0 rad/s^2 per joint)
    a.add_argument('--velocity', type=float, default=0.6,
                   help='velocity scaling for the free moves (0.6 = 17 deg/s)')
    a.add_argument('--slow-velocity', type=float, default=0.25,
                   help='velocity scaling near the desk: grasp, put down, lift (0.25 = 7 deg/s)')
    a.add_argument('--grasp-velocity', type=float, default=None,
                   help='velocity scaling of the pick descent only (default: --slow-velocity)')
    a.add_argument('--refine', action='store_true',
                   help='pick: detect the cube again from above it and re-plan the descent')
    a.add_argument('--refine-settle', type=float, default=0.5,
                   help='refine: wait after the gripper opened before detecting [s]')
    a.add_argument('--refine-min', type=float, default=0.001,
                   help='refine: re-plan only if the cube moved more than this [m]')
    a.add_argument('--refine-max', type=float, default=0.02,
                   help='refine: stop if it moved more than this (wrong detection) [m]')
    a.add_argument('--accel', type=float, default=0.5, help='acceleration scaling')
    a.add_argument('--search', action='store_true',
                   help='go to the search pose first (it becomes home), then --do / detect')
    a.add_argument('--search-x', type=float, default=0.25,
                   help='search pose: desk point straight in front of the base, x [m]')
    a.add_argument('--search-y', type=float, default=0.0, help='search pose: y [m]')
    a.add_argument('--search-height', type=float, default=0.28,
                   help='search pose: camera above the desk [m] (lower tried if unreachable)')
    a.add_argument('--search-min-height', type=float, default=0.18)
    a.add_argument('--no-return-home', dest='return_home', action='store_false',
                   help='stay above the cube / place point instead of going home')
    a.add_argument('--do', choices=['', 'hover', 'pick', 'pickplace'], default='',
                   help='run this at start (pickplace needs --place)')
    a.add_argument('--place', default='', help='x,y place point in base [m] for --do pickplace')
    a.add_argument('--go', action='store_true', help='really move (else plan only)')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    a.add_argument('--no-show', action='store_true', help='no camera window')
    a.add_argument('--holding', action='store_true',
                   help='start as holding the cube (after a restart with the cube in the gripper)')
    args, _ = a.parse_known_args()             # ignore --ros-args from ros2 run
    if args.grasp_z <= args.table_margin:
        raise SystemExit(f'--grasp-z {args.grasp_z} must be above --table-margin '
                         f'{args.table_margin} (MoveIt keeps the fingers above it)')
    if args.do == 'pickplace' and not args.place:
        raise SystemExit('--do pickplace needs --place x,y')
    place = [float(v) for v in args.place.split(',')] if args.place else None
    show = not args.no_show
    if show and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        print('no display: running without the camera window')
        show = False
    if not show and not args.do:
        args.do = 'detect'

    rclpy.init()
    node = Cube(args.image_topic)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        app = App(node, args)
        app.say('ready' + ('' if args.go else ' (DRY RUN: add --go to move)'))
        if args.holding and not args.do:
            args.do = 'none'
        first = {'detect': ('detect', app.act_detect), 'hover': ('hover', app.act_hover),
                 'pick': ('pick', app.act_pick),
                 'pickplace': ('pick and place', app.act_pickplace, place)}
        if args.search:
            name, fn, *rest = first.get(args.do, ('detect', None))
            app.submit('search pose', app.act_search,
                       None if fn is None else (lambda: fn(*rest)))
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
