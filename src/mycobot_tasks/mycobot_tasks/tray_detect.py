"""Black tray + grip pin: YOLO (ONNX, cv2.dnn) finds them, geometry gives the pose.

Needs the camera (/camera/image_raw) and TF base -> camera_optical_frame (e.g. handeye.launch.py)
with the arm in a look-down pose. Detection only, the robot is not moved. Run in the ROS
terminal (not the yolo venv):

  ros2 run mycobot_tasks tray_detect
  ros2 run mycobot_tasks tray_detect --model ~/.../best.onnx --table-z -0.007

Pipeline
  1. YOLO boxes 'tray' and 'pin' (ONNX of tray_yolo/runs/tray/weights/best.pt).
  2. Inside the tray box the black body is segmented (HSV value < --dark-v; the robot shadow
     is brighter, V ~80, the tray V < 20) -> minAreaRect: precise centre and long axis.
  3. The 180 deg ambiguity of the axis: the YOLO pin box says which short end carries the pin.
  4. Pixel -> base: camera ray through the rect centre hits the plane at the rim height
     (--table-z + --tray-h). The pin pose is NOT taken from its (loose) YOLO box but from the
     tray geometry: centre + R(yaw) * (-(--tray-len/2 + --pin-out/2), 0) at --pin-z.
Published (frame 'base', yaw about +z, tray +x points AWAY from the pin; smoothed):
  /tray/pose      PoseStamped  tray centre (at the rim height)
  /tray/pin_pose  PoseStamped  centre of the graspable part of the pin
Nothing is published when the tray touches the image border (partly seen) or no TF.
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
from mycobot_tasks.yolo_onnx import OnnxYolo

BASE = 'base'
DEFAULT_MODEL = os.path.expanduser(
    '~/final_project_ws/src/tray_yolo/runs/tray/weights/best.onnx')


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def refine_tray(bgr, box, dark_v, pad=12):
    """Black body inside the YOLO tray box -> (corners 4x2 px, mask, touches_border) or None."""
    h, w = bgr.shape[:2]
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    roi = bgr[y0:y1, x0:x1]
    m = (roi.max(axis=2) < dark_v).astype(np.uint8) * 255
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
    corners = cv2.boxPoints(cv2.minAreaRect(c)) + np.array([x0, y0], np.float32)
    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = m
    return corners, full, border


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
        self.K, self.D, _ = load_intrinsics(a.camera_calibration_file)
        self.yolo = OnnxYolo(os.path.expanduser(a.model), ['tray', 'pin'], a.imgsz, a.conf)
        self.hist = collections.deque(maxlen=a.smooth)
        self.msg = ''

    def _on_img(self, msg):
        self.img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def T(self):
        try:
            return transform_msg_to_T(self.tf.lookup_transform(
                BASE, CAM, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def process(self, bgr, T, vis=None):
        """-> (centre xy, yaw, pin_xy) in base or None (self.msg says why)."""
        a = self.a
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
        r = refine_tray(bgr, best['tray'][1], a.dark_v)
        if r is None:
            self.msg = 'tray mask failed (check --dark-v)'
            return None
        corners, mask, border = r
        self.mask = mask
        if vis is not None:
            cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 2)
        if border:
            self.msg = 'tray touches the image border: not used'
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
        pin = pc[:2] + d * np.array([-math.cos(ang), -math.sin(ang)])
        self.msg = ''
        return pc, ang, pin

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
        self.mask = None
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
                res = self.process(bgr, T, vis)
                if res is None:
                    self.hist.clear()
                    text = self.msg
                else:
                    pc, ang, pin = self.smooth(*res)
                    self.publish(pc, ang, pin)
                    text = (f'tray ({pc[0]:.3f}, {pc[1]:.3f}) yaw {math.degrees(ang):.1f} deg  '
                            f'pin ({pin[0]:.3f}, {pin[1]:.3f})')
            if text != last and not text.startswith('tray ('):
                self.get_logger().info(text)
            last = text
            if show:
                if view_mask and self.mask is not None:
                    vis = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('tray_detect', vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == ord('m'):
                    view_mask = not view_mask
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--model', default=DEFAULT_MODEL, help='best.onnx of the YOLO model')
    p.add_argument('--conf', type=float, default=0.4)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--dark-v', type=int, default=45,
                   help='HSV value below this = tray black (robot shadow is ~80, tray < 20)')
    p.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    p.add_argument('--tray-h', type=float, default=0.056,
                   help='height of the tray rim above the desk [m] (frame 6 + wall 50 mm)')
    p.add_argument('--tray-len', type=float, default=0.150, help='tray length along its x [m]')
    p.add_argument('--pin-out', type=float, default=0.012, help='pin protrusion [m]')
    p.add_argument('--pin-h', type=float, default=0.045,
                   help='pin top above the desk [m]')
    p.add_argument('--grasp-h', type=float, default=0.028,
                   help='height of the graspable upper part of the pin [m] (-> 17..45 mm)')
    p.add_argument('--smooth', type=int, default=5, help='frames averaged')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    p.add_argument('--no-show', action='store_true')
    a, ros = p.parse_known_args()
    a.pin_z = a.pin_h - a.grasp_h / 2          # middle of the graspable part above the desk
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
