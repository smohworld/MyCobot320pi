"""Distance to the black tray seen from the front (short side facing the camera), no YOLO,
and an approach of the finger tip (gripper_tcp) to a set distance in front of the pin.

Use with the arm in a front view pose (look_toward, e.g. --aim-dz 0.028 for the tray's mid
height). Needs the camera and TF base -> camera_optical_frame (e.g. handeye.launch.py); for
the approach also move_group and the driver.

  ros2 run mycobot_tasks tray_range                        # measure; approach = plan only
  ros2 run mycobot_tasks tray_range --go --real            # key a moves the REAL robot
  ros2 run mycobot_tasks tray_range --go --real --approach 0.02

Distance: the camera has no depth, it comes from the known size of the face (--face-w x
--face-h, default the short side 100 x 56 mm incl. the 6 mm frame). The black blob (HSV
value < --dark-v, gripper fingers cut out from TF) is undistorted to normalized image
coordinates; its extent gives
  Z_w = face_w / width,   Z_h = face_h / height      (distance along the optical axis)
Z_w is used. They disagree when part of a side wall (tray off the image centre / turned) or
the top / inside (camera above the rim) is seen, or the blob is cut: then the box turns red
and nothing is published / approached. A white pin in front of the face leaves a notch in
the blob; the outer extent is unchanged.

Shown (window top left, and printed once per second):
  Z        camera -> face along the optical axis (h: the same from the face height)
  tip      finger tip (gripper_tcp) -> face / -> pin front (face - --pin-out), along the
           horizontal view direction; lat / dz: tip off the pin centre line sideways / in
           height (the grasp height, tray bottom + --grasp-z)
  bottom   face bottom vs the shelf top (--table-z + --shelf-height): ~0 if Z is right

Approach (key a): the tip goes to --approach in front of the pin front, on the pin centre
line (sideways) at the grasp height, keeping the current TCP orientation, measured from the
smoothed detection at the key press. Keys + / - change --approach by 5 mm (not below
--min-approach). b = back to the pose at start, x / SPACE = stop. Without --go: plan only.
After a move the distance shown is measured again (camera), compare it with --approach.

Published (frame base, smoothed over --smooth frames):
  /tray/front_pose  PoseStamped  centre of the facing side, yaw = horizontal view direction
  /tray/pose        PoseStamped  tray centre = face centre + --tray-depth / 2 along the view
                                 (assumes the face is square to the camera)  [--publish-tray]
Window: left click = use the blob under / nearest the click (else the biggest one near the
centre), right click = automatic again, m = mask view, q / ESC = quit.
"""
import argparse
import collections
import math
import os
import queue
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import make_T, transform_msg_to_T
from mycobot_calibration.goto_lookdown import driver_connection
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import CAM, TCP, _text
from mycobot_tasks.tray_detect import fingers_from_tf

BASE = 'base'


def dark_mask(bgr, dark_v, ignore=None):
    m = (bgr.max(axis=2) < dark_v).astype(np.uint8) * 255
    if ignore is not None:
        m[ignore > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def pick_blob(mask, a, click=None):
    """Contour of the tray face: the one under / nearest the click, else the biggest with a
    plausible aspect ratio, preferring the image centre. None if nothing fits."""
    h, w = mask.shape
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    want = a.face_w / a.face_h
    best = None
    for c in cs:
        area = cv2.contourArea(c)
        if area < a.min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if click is not None:
            d = cv2.pointPolygonTest(c, (float(click[0]), float(click[1])), True)
            score = -max(0.0, -d)                  # inside: 0, outside: -distance
        else:
            ratio = bw / max(bh, 1)
            if not (want / a.aspect_tol <= ratio <= want * a.aspect_tol):
                continue
            off = math.hypot(x + bw / 2 - w / 2, y + bh / 2 - h / 2) / math.hypot(w, h)
            score = area * (1.0 - off)
        if best is None or score > best[0]:
            best = (score, c)
    return None if best is None else best[1]


def measure(c, K, D, a, shape):
    """Contour -> dict(Zw, Zh, centre normalized (x, y), box px, cut) using undistorted
    extents (outer extent, so a pin notch does not matter)."""
    pts = c.reshape(-1, 1, 2).astype(np.float64)
    n = cv2.undistortPoints(pts, K, D).reshape(-1, 2)
    x0, y0 = n.min(axis=0)
    x1, y1 = n.max(axis=0)
    bx, by, bw, bh = cv2.boundingRect(c)
    h, w = shape[:2]
    cut = bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2
    return dict(Zw=a.face_w / (x1 - x0), Zh=a.face_h / (y1 - y0),
                cn=((x0 + x1) / 2, (y0 + y1) / 2), box=(bx, by, bw, bh), cut=cut)


def tip_vs_pin(T_tcp, face, v, a):
    """Finger tip relative to the pin: (d_face, d_pin, lateral, dz) [m]. Distances along the
    horizontal view direction v; lateral positive = pin is to the left of the tip (seen
    along v); dz = pin grasp height - tip height."""
    tip = T_tcp[:3, 3]
    d_face = float((face - tip) @ v)
    side = np.array([-v[1], v[0], 0.0])
    grasp_z = face[2] - a.face_h / 2 + a.grasp_z
    return d_face, d_face - a.pin_out, float((face - tip) @ side), float(grasp_z - tip[2])


class Stop(Exception):
    pass


class TrayRange(Node):
    def __init__(self, a):
        super().__init__('tray_range')
        self.a = a
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.img = None
        self.js = {}
        cb = ReentrantCallbackGroup()
        self.create_subscription(Image, a.image_topic, self._on_img, 2, callback_group=cb)
        self.create_subscription(JointState, 'joint_states',
                                 lambda m: self.js.update(zip(m.name, m.position)), 10,
                                 callback_group=cb)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub_front = self.create_publisher(PoseStamped, '/tray/front_pose', 5)
        self.pub_tray = self.create_publisher(PoseStamped, '/tray/pose', 5) \
            if a.publish_tray else None
        self.K, self.D, _ = load_intrinsics(a.camera_calibration_file)
        self.hist = collections.deque(maxlen=a.smooth)
        self.click = None
        self.mask = None
        self.det = None             # last good smoothed detection: dict(face, v, t)

    def _on_img(self, msg):
        with self.lock:
            self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def take_img(self):
        with self.lock:
            img, self.img = self.img, None
        return img

    def T(self, parent=BASE, child=CAM):
        try:
            return transform_msg_to_T(self.tf.lookup_transform(
                parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def q(self):
        return np.array([self.js[n] for n in ARM_JOINTS]) if all(
            n in self.js for n in ARM_JOINTS) else None

    def _pose(self, p, yaw):
        m = PoseStamped()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), BASE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
        m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m

    def step(self, bgr, T, vis):
        """One frame -> (status lines, ok)."""
        a = self.a
        mask = dark_mask(bgr, a.dark_v, fingers_from_tf(self.T, bgr.shape, self.K))
        self.mask = mask
        c = pick_blob(mask, a, self.click)
        if c is None:
            self.hist.clear()
            self.det = None
            return ['no tray face (dark blob with the right aspect ratio)'], False
        r = measure(c, self.K, self.D, a, bgr.shape)
        p_cam = r['Zw'] * np.array([r['cn'][0], r['cn'][1], 1.0])  # face centre, camera
        p = T[:3, :3] @ p_cam + T[:3, 3]
        v = T[:3, 2].copy()                                      # optical axis in base
        v[2] = 0.0
        v /= max(np.linalg.norm(v), 1e-9)
        cur = np.array([p[0], p[1], p[2], r['Zw'], r['Zh'], v[0], v[1]])
        if self.hist and np.linalg.norm(cur[:3] - np.mean(self.hist, axis=0)[:3]) > 0.03:
            self.hist.clear()
        self.hist.append(cur)
        m = np.mean(self.hist, axis=0)
        p, Z, Zh = m[:3], m[3], m[4]
        v = np.array([m[5], m[6], 0.0])
        v /= np.linalg.norm(v)
        yaw = math.atan2(v[1], v[0])
        dz_bottom = (p[2] - a.face_h / 2) - (a.table_z + a.shelf_height)
        agree = abs(Z - Zh) / Z
        ok = agree < a.max_disagree and not r['cut']
        self.det = dict(face=p, v=v, t=time.monotonic()) if ok else None
        if ok:
            self.pub_front.publish(self._pose(p, yaw))
            if self.pub_tray is not None:
                self.pub_tray.publish(self._pose(p + v * a.tray_depth / 2, yaw))
        if vis is not None:
            bx, by, bw, bh = r['box']
            cv2.drawContours(vis, [c], -1, (255, 0, 255), 1)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0) if ok else
                          (0, 0, 255), 2)
        why = ' CUT by the image border' if r['cut'] else (
            f' w/h DISAGREE {agree * 100:.0f}%' if agree >= a.max_disagree else '')
        lines = [f'Z {Z * 1000:.0f} mm (h: {Zh * 1000:.0f})  face ({p[0] * 1000:.0f}, '
                 f'{p[1] * 1000:.0f}, {p[2] * 1000:.0f}) mm  bottom vs shelf '
                 f'{dz_bottom * 1000:+.0f} mm{why}']
        T_tcp = self.T(BASE, TCP)
        if T_tcp is not None:
            d_face, d_pin, lat, dz = tip_vs_pin(T_tcp, p, v, a)
            lines.append(f'tip -> face {d_face * 1000:.0f} mm  -> pin {d_pin * 1000:.0f} mm  '
                         f'(lat {lat * 1000:+.0f}, dz {dz * 1000:+.0f} mm)  '
                         f'approach to {a.approach * 1000:.0f} mm')
        return lines, ok


class App:
    """Approach / back moves in a worker thread (window and detection keep running)."""

    def __init__(self, node, a):
        self.node, self.a = node, a
        self.status = ''
        self.busy = False
        self.jobs = queue.Queue()
        self._abort = threading.Event()
        self.mv = None
        self.home = None
        self.conn = driver_connection(node)
        if a.go and self.conn != 'fake' and not a.real:
            raise SystemExit(f'driver connection is {self.conn!r}, not fake: add --real '
                             'to move it')
        self.mv = MoveItClient(node)
        if not self.mv.wait_ready(10.0):
            print('move_group not available: measuring only (keys a / b disabled)')
            self.mv = None
        t0 = time.monotonic()
        while node.q() is None and time.monotonic() - t0 < 5.0:
            time.sleep(0.1)
        self.home = node.q()
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, s):
        self.status = s
        print(s, flush=True)

    def submit(self, name, fn):
        if self.mv is None:
            self.say('no move_group: cannot move')
            return
        if self.busy:
            self.say(f'busy - {name} ignored (x / SPACE stops)')
            return
        self.busy = True
        self._abort.clear()
        self.jobs.put((name, fn))

    def _worker(self):
        while True:
            name, fn = self.jobs.get()
            try:
                fn()
            except Stop as e:
                self.say(f'{name}: {e}')
            except Exception as e:  # noqa: BLE001
                self.say(f'{name}: ERROR {e!r}')
            finally:
                self.busy = False

    def stop(self):
        self._abort.set()
        if self.mv is not None:
            self.mv.cancel()
        self.say('STOP requested')

    def move_q(self, name, q):
        print(f'  {name} J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
        if self._abort.is_set():
            raise Stop('stopped')
        if not self.a.go:
            self.say(f'(dry run) {name}: add --go to move')
            return False
        self.say(f'moving: {name}')
        ok, msg = self.mv.move_joints(ARM_JOINTS, q, self.a.velocity, self.a.accel)
        if not ok:
            raise Stop(f'motion failed ({msg})')
        return True

    def act_approach(self):
        a, node = self.a, self.node
        det = node.det
        if det is None or time.monotonic() - det['t'] > 1.0:
            raise Stop('no valid tray detection (green box) right now')
        T_tcp, q0 = node.T(BASE, TCP), node.q()
        if T_tcp is None or q0 is None:
            raise Stop(f'no TF base -> {TCP} / joint_states')
        face, v = det['face'], det['v']
        d_face, d_pin, lat, dz = tip_vs_pin(T_tcp, face, v, a)
        tilt = math.degrees(math.acos(np.clip(T_tcp[:3, 2] @ v, -1, 1)))
        if tilt > a.max_tilt:
            raise Stop(f'TCP z axis is {tilt:.0f} deg off the view direction (> --max-tilt): '
                       'not a front pose (run look_toward first)')
        if d_pin < a.approach - 1e-4 and abs(lat) < 0.005 and abs(dz) < 0.005:
            raise Stop(f'tip already {d_pin * 1000:.0f} mm from the pin (< --approach)')
        side = np.array([-v[1], v[0], 0.0])
        goal = T_tcp[:3, 3] + v * (d_pin - a.approach) + side * lat + np.array([0, 0, dz])
        T_goal = make_T(T_tcp[:3, :3], goal)
        q = self.mv.compute_ik(T_goal, TCP, ARM_JOINTS, q0, timeout=0.2)
        if q is None:
            raise Stop(f'no IK for the approach ({goal[0] * 1000:.0f}, {goal[1] * 1000:.0f}, '
                       f'{goal[2] * 1000:.0f}) mm')
        q = np.asarray(q)
        jump = math.degrees(float(np.max(np.abs(q - q0))))
        if jump > a.max_joint_jump:
            raise Stop(f'IK solution jumps {jump:.0f} deg (> --max-joint-jump): refused')
        self.say(f'approach: tip -> pin {d_pin * 1000:.0f} -> {a.approach * 1000:.0f} mm, '
                 f'side {lat * 1000:+.0f}, up {dz * 1000:+.0f} mm')
        if self.move_q('approach', q):
            time.sleep(0.5)
            T = node.T(BASE, TCP)
            if T is not None:
                e = goal - T[:3, 3]
                self.say(f'approach done: tip vs goal (model) {np.linalg.norm(e) * 1000:.1f} '
                         'mm; see the camera distance for the real one')

    def act_back(self):
        if self.home is None:
            raise Stop('no start pose recorded')
        if self.move_q('back to start', self.home):
            self.say('back at the start pose')


def run(node, app):
    a = node.a
    show, view_mask, last_print = not a.no_show, False, 0.0
    win = 'tray_range'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        def on_mouse(ev, u, v, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN:
                node.click = (u, v)
                node.hist.clear()
            elif ev == cv2.EVENT_RBUTTONDOWN:
                node.click = None
                node.hist.clear()
        cv2.setMouseCallback(win, on_mouse)
    while rclpy.ok():
        bgr = node.take_img()
        if bgr is None:
            time.sleep(0.01)
            if show and cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            continue
        vis = bgr.copy() if show else None
        T = node.T()
        if T is None:
            lines, ok = [f'no TF {BASE} -> {CAM}'], False
        else:
            lines, ok = node.step(bgr, T, vis)
        now = time.monotonic()
        if now - last_print > 1.0:
            print('  |  '.join(lines), flush=True)
            last_print = now
        if not show:
            continue
        if view_mask and node.mask is not None:
            vis = cv2.cvtColor(node.mask, cv2.COLOR_GRAY2BGR)
        if node.click is not None:
            cv2.drawMarker(vis, node.click, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        for i, s in enumerate(lines + [('BUSY  ' if app.busy else '') + app.status]):
            _text(vis, s, (8, 20 + 20 * i), 0.45)
        _text(vis, f"{'' if a.go else 'DRY RUN  '}a approach  +/- 5 mm  b back  x stop  "
                   'L/R-click blob  m mask  q quit', (8, vis.shape[0] - 10), 0.42)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('m'):
            view_mask = not view_mask
        elif k in (ord('x'), ord(' ')):
            app.stop()
        elif k == ord('a'):
            app.submit('approach', app.act_approach)
        elif k == ord('b'):
            app.submit('back', app.act_back)
        elif k in (ord('+'), ord('=')):
            a.approach += 0.005
            app.say(f'approach distance {a.approach * 1000:.0f} mm')
        elif k in (ord('-'), ord('_')):
            a.approach = max(a.min_approach, a.approach - 0.005)
            app.say(f'approach distance {a.approach * 1000:.0f} mm')
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--face-w', type=float, default=0.100,
                   help='width of the side facing the camera [m] (short 0.100, long 0.150)')
    p.add_argument('--face-h', type=float, default=0.056,
                   help='height of that side [m] (frame 6 + wall 50 mm)')
    p.add_argument('--pin-out', type=float, default=0.015,
                   help='pin protrusion in front of the face [m]')
    p.add_argument('--grasp-z', type=float, default=0.031,
                   help='approach height above the tray bottom [m] (middle of the graspable '
                        'part of the pin, 17..45 mm)')
    p.add_argument('--approach', type=float, default=0.03,
                   help='key a: finger tip this far in front of the pin front [m]')
    p.add_argument('--min-approach', type=float, default=0.0,
                   help='keys +/-: lowest --approach [m]')
    p.add_argument('--max-tilt', type=float, default=15.0,
                   help='approach only if the TCP z axis is within this of the view [deg]')
    p.add_argument('--max-joint-jump', type=float, default=30.0,
                   help='refuse an approach IK solution moving any joint more [deg]')
    p.add_argument('--tray-depth', type=float, default=0.150,
                   help='tray size along the view (for /tray/pose) [m]')
    p.add_argument('--publish-tray', action='store_true',
                   help='also publish /tray/pose (tray centre); off: tray_detect uses it')
    p.add_argument('--dark-v', type=int, default=45, help='BGR max below this = tray black')
    p.add_argument('--min-area', type=float, default=400.0, help='blob area [px^2]')
    p.add_argument('--aspect-tol', type=float, default=1.6,
                   help='auto pick: blob w/h within face_w/face_h */ this')
    p.add_argument('--max-disagree', type=float, default=0.08,
                   help='Z_w vs Z_h relative difference above this = not used')
    p.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    p.add_argument('--shelf-height', type=float, default=0.215,
                   help='shelf (book) top over the desk [m], only for the bottom check')
    p.add_argument('--smooth', type=int, default=5, help='frames averaged')
    p.add_argument('--velocity', type=float, default=0.15, help='velocity scaling')
    p.add_argument('--accel', type=float, default=0.15, help='acceleration scaling')
    p.add_argument('--go', action='store_true', help='really move (else plan only)')
    p.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    p.add_argument('--no-show', action='store_true', help='no window (measure only)')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    a, ros = p.parse_known_args()
    rclpy.init(args=ros)
    node = TrayRange(a)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        app = App(node, a)
        run(node, app)
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
