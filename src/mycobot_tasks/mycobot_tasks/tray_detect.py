"""Black tray + grip pin: YOLO (ONNX, cv2.dnn) finds them, geometry gives the pose.

Needs the camera (/camera/image_raw) and TF base -> camera_optical_frame (e.g. handeye.launch.py)
with the arm in a look-down pose. Detection only, the robot is not moved. Run in the ROS
terminal (not the yolo venv):

  ros2 run mycobot_tasks tray_detect
  ros2 run mycobot_tasks tray_detect --model ~/.../best.onnx --table-z -0.007

Pipeline
  1. YOLO boxes 'tray' and 'pin' (ONNX of tray_yolo/runs/tray_auto/weights/best.pt; the
     'pin' box is the tan tape on the pin top, see tray_yolo/autolabel.py).
  2. Inside the tray box the black body is segmented (HSV value < --dark-v; the robot shadow
     is brighter, V ~80, the tray V < 20) -> minAreaRect: precise centre and long axis.
  3. The 180 deg ambiguity of the axis: the YOLO pin box says which short end carries the pin.
  4. Pixel -> base: camera ray through the rect centre hits the plane at the rim height
     (--table-z + --tray-h).
  5. Pin xy, --pin-source tape (default): camera ray through the tape box centre hits the
     plane of the pin top (--table-z + --pin-h). --pin-source geometry: tray centre +
     R(yaw) * (-(--tray-len/2 + --pin-out/2), 0). Both are computed; when they differ by more
     than --max-pin-diff the frame is refused (wrong box?). Window: yellow = tape, magenta =
     geometry.
Published (frame 'base', yaw about +z, tray +x points AWAY from the pin; smoothed):
  /tray/pose      PoseStamped  tray centre (at the rim height)
  /tray/pin_pose  PoseStamped  centre of the graspable part of the pin
Nothing is published when the tray touches the image border or the gripper fingers (partly
seen; the fingers are black too and are cut out of the mask, from TF) or there is no TF.
Window: m = mask view, q / ESC quit.
"""
import argparse
import collections
import math
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import transform_msg_to_T
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_tasks.red_cube_hover import CAM, ray_plane
from mycobot_tasks.yolo_remote import CLASSES, make_yolo

BASE = 'base'
DEFAULT_MODEL = os.path.expanduser(
    '~/final_project_ws/src/tray_yolo/runs/tray_auto/weights/best.onnx')


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def project_xy(xy, z, T_bc, K, D):
    """Base point (x, y, z) -> pixel (int u, v) or None when behind the camera."""
    p = np.linalg.inv(T_bc) @ np.array([xy[0], xy[1], z, 1.0])
    if p[2] <= 0:
        return None
    uv, _ = cv2.projectPoints(p[:3].reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    return tuple(int(round(v)) for v in uv.ravel())


def finger_mask(shape, T_cam_tcp, K, x_left, x_right, side=0.012, thick=0.008, length=0.06,
                pad=8):
    """Image region of the gripper fingers (they are black like the tray): per finger a box in
    the TCP frame from the finger tip (TCP height) back by `length`, `side` around the finger
    x position, projected without distortion (points this close to the camera break the
    distortion model) and dilated by `pad` px. The two fingers are separate regions: the gap
    between open fingers (where the tray / pin is seen) stays free."""
    m = np.zeros(shape[:2], np.uint8)
    for xf in (x_left, x_right):
        pts = np.array([[x, y, z, 1.0] for x in (xf - side, xf + side)
                        for y in (-thick, thick) for z in (-length, 0.005)]).T
        pc = (T_cam_tcp @ pts)[:3].T
        pc = pc[pc[:, 2] > 0.01]
        if len(pc) >= 3:
            uv = (K @ pc.T).T
            uv = np.clip(uv[:, :2] / uv[:, 2:3], -5000, 5000)
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 255)
    if pad:
        m = cv2.dilate(m, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    return m


def fingers_from_tf(lookup, shape, K, cam='camera_optical_frame', tcp='gripper_tcp'):
    """finger_mask from TF (lookup(parent, child) -> 4x4 or None); None without TF."""
    T = lookup(cam, tcp)
    xl, xr = lookup(tcp, 'gripper_left1'), lookup(tcp, 'gripper_right1')
    if T is None or xl is None or xr is None:
        return None
    a, b = sorted((xl[0, 3], xr[0, 3]))
    return finger_mask(shape, T, K, a, b)


def refine_tray(bgr, box, dark_v, pad=12, ignore=None):
    """Black body inside the YOLO tray box -> (corners 4x2 px, mask, touches_border) or None.
    `ignore` (e.g. the gripper fingers) is cut out of the mask; a blob that touches it counts
    as touching the border (the tray may continue under it)."""
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    roi = bgr[y0:y1, x0:x1]
    m = (roi.max(axis=2) < dark_v).astype(np.uint8) * 255
    if ignore is not None:
        m[ignore[y0:y1, x0:x1] > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.25 * (box[2] - box[0]) * (box[3] - box[1]):
        return None
    bx, by, bw, bh = cv2.boundingRect(c)
    border = (bx + x0 <= 2 or by + y0 <= 2 or bx + bw + x0 >= w - 2 or by + bh + y0 >= h - 2)
    if ignore is not None and not border:
        blob = np.zeros_like(m)
        cv2.drawContours(blob, [c], -1, 255, -1)
        blob = cv2.dilate(blob, np.ones((7, 7), np.uint8))
        border = bool(np.any((blob > 0) & (ignore[y0:y1, x0:x1] > 0)))
    corners = cv2.boxPoints(cv2.minAreaRect(c)) + np.array([x0, y0], np.float32)
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = m
    return corners, full, border


class TrayDetector:
    """The detection itself (no ROS): used by the tray_detect node and inside tray_pick."""

    def __init__(self, a, K, D):
        self.a, self.K, self.D = a, K, D
        self.yolo = make_yolo(a.model, CLASSES, a.imgsz, a.conf, a.yolo_sock)
        self.msg = ''
        self.pin_diff = 0.0
        self.mask = None
        self.partial = None

    def wall_axis(self, mask, px, T, ignore=None, vis=None, radius=70, min_pts=30):
        """Direction (base, rad, mod 180 deg) of the tray wall the pin sits on: a line fitted to
        the black-mask boundary within `radius` px of the pin pixel (image border and finger
        pixels left out), or None."""
        h, w = mask.shape[:2]
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        pts = np.vstack([c.reshape(-1, 2) for c in cs]).astype(np.float64)
        keep = np.hypot(pts[:, 0] - px[0], pts[:, 1] - px[1]) < radius
        keep &= (pts[:, 0] > 3) & (pts[:, 1] > 3) & (pts[:, 0] < w - 4) & (pts[:, 1] < h - 4)
        if ignore is not None:
            near = cv2.dilate(ignore, np.ones((9, 9), np.uint8))
            keep &= near[pts[:, 1].astype(int), pts[:, 0].astype(int)] == 0
        pts = pts[keep]
        if len(pts) < min_pts:
            return None
        n_all = len(pts)
        for tol in (8.0, 4.0, 3.0):          # refit on the points near the line (inliers):
            vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01,
                                         0.01).ravel()
            res = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
            pts = pts[res < tol]             # mask steps, cut edges, the pin mount drop out
            if len(pts) < min_pts:
                return None
        if len(pts) < 0.3 * n_all:           # no dominant straight edge (e.g. a corner)
            return None
        if vis is not None:
            d = np.array([vx, vy]) * radius
            cv2.line(vis, tuple(np.int32([x0, y0] - d)), tuple(np.int32([x0, y0] + d)),
                     (255, 255, 0), 2)
        z = self.a.table_z + self.a.tray_h
        p0 = ray_plane(np.array([x0, y0]), self.K, self.D, T, z)[0]
        p1 = ray_plane(np.array([x0 + 20 * vx, y0 + 20 * vy]), self.K, self.D, T, z)[0]
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def tape_axis(self, bgr, px, T):
        """Direction (base, rad, mod 90 deg) of an edge of the OBB 'pin' box nearest pixel px."""
        try:
            obbs = [d for d in self.yolo.detect_obb(bgr) if d[0] == 'pin']
        except ValueError:                           # not an OBB model
            return None
        if not obbs:
            return None
        d = min(obbs, key=lambda d: np.hypot(d[2] - px[0], d[3] - px[1]))
        e = d[7][1] - d[7][0]
        e = e / max(np.linalg.norm(e), 1e-6)
        z = self.a.table_z + self.a.pin_h
        p0 = ray_plane(px, self.K, self.D, T, z)[0]
        p1 = ray_plane(px + 20.0 * e, self.K, self.D, T, z)[0]
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def pin_only(self, bgr, T):
        """Tape (pin top) xy in base from the best YOLO 'pin' box alone, or None. For the
        close-up servo above the pin, where the tray is usually not fully in view."""
        best = None
        for n, c, *b in self.yolo.detect(bgr):
            if n == 'pin' and (best is None or c > best[0]):
                best = (c, b)
        if best is None:
            return None
        b = best[1]
        return ray_plane(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), self.K, self.D, T,
                         self.a.table_z + self.a.pin_h)[0][:2]

    def process(self, bgr, T, vis=None, ignore=None):
        """-> (centre xyz, yaw, pin_xy) in base or None (self.msg says why; self.partial =
        dict(blob=xy of the visible part, pin=xy of the tape or None, axis=direction of one tape
        edge in base [rad, mod 90 deg] or None) when the tray touches the image border or the
        fingers)."""
        a = self.a
        self.partial = None
        dets = self.yolo.detect(bgr)
        best = {}
        for n, c, *b in dets:
            if n not in best or c > best[n][0]:
                best[n] = (c, b)
        if vis is not None:
            for n, (c, b) in best.items():
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                              (0, 255, 0) if n == 'tray' else (0, 165, 255), 1)
        if 'tray' not in best:
            self.msg = 'no tray'
            return None
        if vis is not None and ignore is not None:
            vis[ignore > 0] = (vis[ignore > 0] * 0.6).astype(np.uint8)
        r = refine_tray(bgr, best['tray'][1], a.dark_v, ignore=ignore)
        if r is None:
            self.msg = 'tray mask failed (check --dark-v)'
            if 'pin' in best:                   # the pin alone is still usable (tray_pick)
                tb = best['tray'][1]
                c = np.array([(tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2])
                pb = best['pin'][1]
                pc_px = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
                self.partial = dict(
                    blob=ray_plane(c, self.K, self.D, T, a.table_z + a.tray_h)[0][:2],
                    pin=ray_plane(pc_px, self.K, self.D, T, a.table_z + a.pin_h)[0][:2],
                    axis=self.tape_axis(bgr, pc_px, T), wall=None)
            return None
        corners, mask, border = r
        self.mask = mask
        if vis is not None:
            cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 2)
        if border:
            mu = cv2.moments(mask, binaryImage=True)
            if mu['m00'] > 0:                   # visible part of the tray -> look there
                c = np.array([mu['m10'] / mu['m00'], mu['m01'] / mu['m00']])
                pin, axis, wall = None, None, None
                if 'pin' in best:
                    pb = best['pin'][1]
                    pc_px = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
                    pin = ray_plane(pc_px, self.K, self.D, T, a.table_z + a.pin_h)[0][:2]
                    axis = self.tape_axis(bgr, pc_px, T)
                    wall = self.wall_axis(mask, pc_px, T, ignore, vis)
                self.partial = dict(
                    blob=ray_plane(c, self.K, self.D, T, a.table_z + a.tray_h)[0][:2], pin=pin,
                    axis=axis, wall=wall)
            self.msg = 'tray touches the image border / gripper fingers: not used'
            return None
        e = [corners[1] - corners[0], corners[2] - corners[1]]
        axis = e[0] if np.linalg.norm(e[0]) >= np.linalg.norm(e[1]) else e[1]
        axis = axis / np.linalg.norm(axis)
        cpx = corners.mean(axis=0)
        z = a.table_z + a.tray_h
        pc = ray_plane(cpx, self.K, self.D, T, z)[0]
        pa = ray_plane(cpx + 60 * axis, self.K, self.D, T, z)[0]
        ang = math.atan2(pa[1] - pc[1], pa[0] - pc[0])               # long axis, mod pi
        if 'pin' not in best:
            self.msg = 'tray found, no pin box: yaw ambiguous (+-180)'
            return None
        pb = best['pin'][1]
        pp = ray_plane(np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2]),
                       self.K, self.D, T, a.table_z + a.pin_z)[0]
        to_pin = pp[:2] - pc[:2]
        if math.cos(ang) * to_pin[0] + math.sin(ang) * to_pin[1] > 0:
            ang = wrap_pi(ang + math.pi)                            # +x points away from pin
        d = a.tray_len / 2 + a.pin_out / 2
        pin_geo = pc[:2] + d * np.array([-math.cos(ang), -math.sin(ang)])
        pin_tape = ray_plane(np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2]),
                             self.K, self.D, T, a.table_z + a.pin_h)[0][:2]
        self.pin_diff = float(np.linalg.norm(pin_tape - pin_geo))
        if vis is not None:
            for q, col in ((pin_tape, (0, 255, 255)), (pin_geo, (255, 0, 255))):
                uv = project_xy(q, a.table_z + a.pin_h, T, self.K, self.D)
                if uv is not None:
                    cv2.drawMarker(vis, uv, col, cv2.MARKER_TILTED_CROSS, 14, 2)
        if self.pin_diff > a.max_pin_diff:
            self.msg = (f'tape and geometry pin differ {self.pin_diff * 1000:.0f} mm '
                        f'> --max-pin-diff: not used')
            return None
        pin = pin_tape if a.pin_source == 'tape' else pin_geo
        self.msg = ''
        return pc, ang, pin



class TrayDetect(Node):
    def __init__(self, a):
        super().__init__('tray_detect')
        self.a = a
        self.bridge = CvBridge()
        self.img = None
        self.create_subscription(Image, a.image_topic, self._on_img, 2)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub_tray = self.create_publisher(PoseStamped, '/tray/pose', 5)
        self.pub_pin = self.create_publisher(PoseStamped, '/tray/pin_pose', 5)
        K, D, _ = load_intrinsics(a.camera_calibration_file)
        self.det = TrayDetector(a, K, D)
        self.hist = collections.deque(maxlen=a.smooth)

    def _on_img(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def T(self, parent=BASE, child=CAM):
        try:
            return transform_msg_to_T(self.tf.lookup_transform(
                parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def smooth(self, pc, ang, pin):
        cur = np.array([pc[0], pc[1], pc[2], math.cos(ang), math.sin(ang), pin[0], pin[1]])
        if self.hist and np.linalg.norm(cur[:2] - np.mean(self.hist, axis=0)[:2]) > 0.03:
            self.hist.clear()                                       # it moved: restart
        self.hist.append(cur)
        m = np.mean(self.hist, axis=0)
        return m[:3], math.atan2(m[4], m[3]), m[5:7]

    def publish(self, pc, ang, pin):
        now = self.get_clock().now().to_msg()
        for pub, p in ((self.pub_tray, pc), (self.pub_pin, (pin[0], pin[1],
                                                           self.a.table_z + self.a.pin_z))):
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = now, BASE
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
            m.pose.orientation.z, m.pose.orientation.w = math.sin(ang / 2), math.cos(ang / 2)
            pub.publish(m)

    def run(self):
        show, view_mask, last = not self.a.no_show, False, ''
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)
            bgr = self.img
            if bgr is None:
                continue
            self.img = None
            vis = bgr.copy() if show else None
            T = self.T()
            text = f'no TF {BASE} -> {CAM}'
            if T is not None:
                res = self.det.process(bgr, T, vis,
                                       fingers_from_tf(self.T, bgr.shape, self.det.K))
                if res is None:
                    self.hist.clear()
                    text = self.det.msg
                else:
                    pc, ang, pin = self.smooth(*res)
                    self.publish(pc, ang, pin)
                    text = (f'tray ({pc[0]:.3f}, {pc[1]:.3f}) yaw {math.degrees(ang):.1f} deg  '
                            f'pin ({pin[0]:.3f}, {pin[1]:.3f}) [{self.a.pin_source}]  '
                            f'tape-geo {self.det.pin_diff * 1000:.1f} mm')
            if text != last and not text.startswith('tray ('):
                self.get_logger().info(text)
            last = text
            if show:
                if view_mask and self.det.mask is not None:
                    vis = cv2.cvtColor(self.det.mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('tray_detect', vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == ord('m'):
                    view_mask = not view_mask
        cv2.destroyAllWindows()


def add_detect_args(p):
    """Detection options, shared by tray_detect and tray_pick."""
    p.add_argument('--model', default=DEFAULT_MODEL, help='best.onnx of the YOLO model')
    p.add_argument('--conf', type=float, default=0.4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--dark-v', type=int, default=45,
                   help='HSV value below this = tray black (robot shadow is ~80, tray < 20)')
    p.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    p.add_argument('--tray-h', type=float, default=0.056,
                   help='height of the tray rim above the desk [m] (frame 6 + wall 50 mm)')
    p.add_argument('--tray-len', type=float, default=0.150, help='tray length along its x [m]')
    p.add_argument('--tray-wid', type=float, default=0.100, help='tray width [m]')
    p.add_argument('--pin-out', type=float, default=0.012, help='pin protrusion [m]')
    p.add_argument('--pin-h', type=float, default=0.045,
                   help='pin top above the desk [m]')
    p.add_argument('--grasp-h', type=float, default=0.028,
                   help='height of the graspable upper part of the pin [m] (-> 17..45 mm)')
    p.add_argument('--pin-source', choices=['tape', 'geometry'], default='tape',
                   help='pin xy from the tape box (YOLO) or from the tray rectangle')
    p.add_argument('--max-pin-diff', type=float, default=0.025,
                   help='refuse when tape and geometry pin xy differ more than this [m]')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    p.add_argument('--yolo-sock', default='/tmp/tray_yolo.sock',
                   help="GPU YOLO server socket (tray_yolo/yolo_server.py); '' = always the "
                        'ONNX model on the CPU; without the server it falls back to the CPU')


def finish_detect_args(a):
    a.pin_z = a.pin_h - a.grasp_h / 2          # middle of the graspable part above the desk
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_detect_args(p)
    p.add_argument('--smooth', type=int, default=5, help='frames averaged')
    p.add_argument('--no-show', action='store_true')
    a, ros = p.parse_known_args()
    finish_detect_args(a)
    rclpy.init(args=ros)
    node = TrayDetect(a)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
