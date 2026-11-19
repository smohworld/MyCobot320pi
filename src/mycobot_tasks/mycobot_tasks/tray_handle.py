"""Tray handle grasp point from the two ArUco markers stuck left / right of the handle
(calibration/tray_aruco_4x4_a4.pdf, DICT_4X4_50, 22.5 mm; tray t = ID 10+2t left,
11+2t right). The grasp point is the midpoint of the two marker centres. Measure only,
the robot is not moved.

Needs the camera and TF base -> camera_optical_frame (e.g. handeye.launch.py).

  ros2 run mycobot_tasks tray_handle                 # tray 0 (IDs 10, 11)
  ros2 run mycobot_tasks tray_handle --gap 0.024     # check vs the real tile-edge gap

Each marker: solvePnP (IPPE_SQUARE) on its 4 corners -> centre and normal in the camera
frame. Midpoint = mean of the two centres, also in base via TF.

Shown (window top left, and printed once per second):
  cam      camera -> midpoint: straight distance and Z (along the optical axis)
  base     midpoint in base [mm], handle yaw (direction L -> R) and tilt of the marker plane
  tip      finger tip (gripper_tcp) -> midpoint: distance and (dx, dy, dz) in base
  L-R      measured distance between the marker centres, and the tile-edge gap (centre
           distance - --tile-size 30 mm) vs --gap 24 mm: a quick
           check of the size / depth (wrong marker size or bad intrinsics -> it is off)
  normals  angle between the two marker normals (~0 if both are flat on the same plane)
  x vs L->R  angle between the marker x axes and the line L -> R: ~0 when both markers
           have their printed top edge towards the same side (inside / front of the tray)

Published (frame base, smoothed over --smooth frames):
  /tray/handle_pose  PoseStamped  midpoint; x = L -> R, z = mean marker normal (outwards)
Keys: m = print the current values once more, q / ESC = quit.
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
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import transform_msg_to_T
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_tasks.red_cube_hover import CAM, TCP, _text

BASE = 'base'
ID_OFFSET = 10


def R_to_quat(R):
    """3x3 rotation -> (x, y, z, w)."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = math.copysign(x, R[2, 1] - R[1, 2])
    y = math.copysign(y, R[0, 2] - R[2, 0])
    z = math.copysign(z, R[1, 0] - R[0, 1])
    return x, y, z, w


class MarkerPose:
    def __init__(self, K, D, size):
        self.K, self.D = K, D
        s = size / 2
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.det = cv2.aruco.ArucoDetector(d, p)

    def detect(self, bgr):
        """-> {id: (corners 4x2 px, rvec, tvec [m], reprojection rms px)}"""
        corners, ids, _ = self.det.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.reshape(-1)):
            img = c.reshape(4, 2).astype(np.float64)
            ok, rv, tv = cv2.solvePnP(self.obj, img, self.K, self.D,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(self.obj, rv, tv, self.K, self.D)
            rms = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, axis=1))))
            out[int(i)] = (img, rv, tv.reshape(3), rms)
        return out


def handle_from_markers(found, id_l, id_r, T_bc, spacing=None):
    """Markers (MarkerPose.detect) + TF base -> camera -> dict(p midpoint xyz in base,
    yaw of L -> R, tilt of the marker plane from horizontal [deg], spacing [m], n markers used)
    or None. With both markers: midpoint of the centres. With one and a known centre spacing:
    its centre +- spacing / 2 along its x axis (both printed tops face the same side, so the
    marker +x points L -> R)."""
    R, t = T_bc[:3, :3], T_bc[:3, 3]
    have = [i for i in (id_l, id_r) if i in found]
    if len(have) < 2 and not (spacing and have):
        return None
    Rm = {i: R @ cv2.Rodrigues(found[i][1])[0] for i in have}
    pc = {i: R @ found[i][2] + t for i in have}
    if len(have) == 2:
        mid = (pc[id_l] + pc[id_r]) / 2
        x = pc[id_r] - pc[id_l]
        sp = float(np.linalg.norm(x))
        z = Rm[id_l][:, 2] + Rm[id_r][:, 2]
    else:
        i = have[0]
        x = Rm[i][:, 0]
        mid = pc[i] + (1 if i == id_l else -1) * x * spacing / 2
        sp = spacing
        z = Rm[i][:, 2]
    z = z / np.linalg.norm(z)
    return dict(p=mid, yaw=math.atan2(x[1], x[0]), spacing=sp, n=len(have), normal=z,
                side=have[0] if len(have) == 1 else None,
                tilt=math.degrees(math.acos(min(1.0, abs(z[2])))))


class TrayHandle(Node):
    def __init__(self, a):
        super().__init__('tray_handle')
        self.a = a
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.img = None
        self.create_subscription(Image, a.image_topic, self._on_img, 2)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub = self.create_publisher(PoseStamped, '/tray/handle_pose', 5)
        self.K, self.D, _ = load_intrinsics(a.camera_calibration_file)
        self.mp = MarkerPose(self.K, self.D, a.marker_size)
        self.id_l = ID_OFFSET + 2 * a.tray
        self.id_r = self.id_l + 1
        self.hist = collections.deque(maxlen=a.smooth)

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

    def step(self, bgr, vis):
        """One frame -> status lines."""
        a = self.a
        found = self.mp.detect(bgr)
        if vis is not None:
            for i, (img, rv, tv, _) in found.items():
                mine = i in (self.id_l, self.id_r)
                cv2.polylines(vis, [img.astype(np.int32)], True,
                              (0, 255, 0) if mine else (128, 128, 128), 2)
                if mine:
                    cv2.drawFrameAxes(vis, self.K, self.D, rv, tv, a.marker_size * 0.6, 2)
                    cv2.putText(vis, 'L' if i == self.id_l else 'R',
                                tuple(img[0].astype(int) + (0, -6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        missing = [n for n, i in (('L', self.id_l), ('R', self.id_r)) if i not in found]
        if missing:
            self.hist.clear()
            others = sorted(set(found) - {self.id_l, self.id_r})
            return [f'tray {a.tray}: marker {"+".join(missing)} (ID '
                    f'{", ".join(str(self.id_l if m == "L" else self.id_r) for m in missing)}) '
                    f'not seen' + (f'  (other IDs seen: {others})' if others else '')]
        _, rvl, pl, el = found[self.id_l]
        _, rvr, pr, er = found[self.id_r]
        Rl, Rr = cv2.Rodrigues(rvl)[0], cv2.Rodrigues(rvr)[0]
        nl, nr = Rl[:, 2], Rr[:, 2]
        # both printed top edges face the same side -> marker +x points L -> R
        u = (pr - pl) / np.linalg.norm(pr - pl)
        xdev = math.degrees(math.acos(np.clip(u @ (Rl[:, 0] + Rr[:, 0]) / np.linalg.norm(
            Rl[:, 0] + Rr[:, 0]), -1, 1)))
        mid_c = (pl + pr) / 2                         # camera frame
        lr_c = pr - pl
        n_c = nl + nr
        n_c /= np.linalg.norm(n_c)
        cur = np.concatenate([mid_c, lr_c, n_c])
        if self.hist and np.linalg.norm(cur[:3] - np.mean(self.hist, axis=0)[:3]) > 0.02:
            self.hist.clear()
        self.hist.append(cur)
        m = np.mean(self.hist, axis=0)
        mid_c, lr_c, n_c = m[:3], m[3:6], m[6:9] / np.linalg.norm(m[6:9])
        spacing = float(np.linalg.norm(lr_c))
        normals = math.degrees(math.acos(np.clip(nl @ nr, -1, 1)))
        lines = [f'cam -> mid {np.linalg.norm(mid_c) * 1000:.1f} mm  (Z {mid_c[2] * 1000:.1f}'
                 f' mm)   L-R {spacing * 1000:.1f} mm'
                 + f' (tile-edge gap {(spacing - a.tile_size) * 1000:.1f}'
                 + (f', real {a.gap * 1000:.1f}: {(spacing - a.tile_size - a.gap) * 1000:+.1f})'
                    if a.gap else ')')
                 + f'  normals {normals:.1f} deg  x vs L->R {xdev:.1f} deg'
                 + (' (L/R SWAPPED or turned?)' if xdev > a.max_xdev else '')
                 + f'  reproj {max(el, er):.2f} px']
        if vis is not None:
            uv, _ = cv2.projectPoints(np.array([pl, pr, (pl + pr) / 2]), np.zeros(3),
                                      np.zeros(3), self.K, self.D)
            uv = uv.reshape(-1, 2).astype(int)
            cv2.line(vis, tuple(uv[0]), tuple(uv[1]), (0, 255, 255), 1)
            cv2.drawMarker(vis, tuple(uv[2]), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        T = self.T()
        if T is None:
            return lines + [f'no TF {BASE} -> {CAM}: base / tip values unavailable']
        R, t = T[:3, :3], T[:3, 3]
        mid = R @ mid_c + t
        x = R @ lr_c
        x /= np.linalg.norm(x)
        z = R @ n_c
        z -= x * (z @ x)
        z /= np.linalg.norm(z)
        Rh = np.column_stack([x, np.cross(z, x), z])
        yaw = math.degrees(math.atan2(x[1], x[0]))
        tilt = math.degrees(math.acos(np.clip(abs(z[2]), -1, 1)))   # 0 = plane horizontal
        lines.append(f'base mid ({mid[0] * 1000:.1f}, {mid[1] * 1000:.1f}, '
                     f'{mid[2] * 1000:.1f}) mm  yaw L->R {yaw:+.1f} deg  '
                     f'plane tilt {tilt:.1f} deg from horizontal')
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), BASE
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, mid)
        (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
         msg.pose.orientation.w) = R_to_quat(Rh)
        self.pub.publish(msg)
        T_tcp = self.T(BASE, TCP)
        if T_tcp is not None:
            d = mid - T_tcp[:3, 3]
            lines.append(f'tip -> mid {np.linalg.norm(d) * 1000:.1f} mm  (dx {d[0] * 1000:+.1f}, '
                         f'dy {d[1] * 1000:+.1f}, dz {d[2] * 1000:+.1f} mm in base)')
        return lines


def run(node):
    a = node.a
    show, last_print, lines = not a.no_show, 0.0, []
    win = 'tray_handle'
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while rclpy.ok():
        bgr = node.take_img()
        if bgr is None:
            time.sleep(0.01)
            if show and cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            continue
        vis = bgr.copy() if show else None
        lines = node.step(bgr, vis)
        now = time.monotonic()
        if now - last_print > 1.0:
            print('  |  '.join(lines), flush=True)
            last_print = now
        if not show:
            continue
        for i, s in enumerate(lines):
            _text(vis, s, (8, 20 + 20 * i), 0.45)
        _text(vis, 'm print  q quit', (8, vis.shape[0] - 10), 0.42)
        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('m'):
            print('\n'.join(lines), flush=True)
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tray', type=int, default=0, help='tray number (IDs 10+2t, 11+2t)')
    p.add_argument('--marker-size', type=float, default=0.0225,
                   help='black marker side [m] (printed 22.5 mm; measure your print)')
    p.add_argument('--gap', type=float, default=0.024,
                   help='real gap between the inner edges of the two cut tiles (white margin '
                        'included) [m] (handle length), for the check')
    p.add_argument('--tile-size', type=float, default=0.030,
                   help='cut marker tile incl. the white margin [m]')
    p.add_argument('--max-xdev', type=float, default=20.0,
                   help='warn when the marker x axes deviate more from L -> R [deg]')
    p.add_argument('--smooth', type=int, default=5, help='frames averaged')
    p.add_argument('--no-show', action='store_true', help='no window (print only)')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    a, ros = p.parse_known_args()
    rclpy.init(args=ros)
    node = TrayHandle(a)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        run(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
