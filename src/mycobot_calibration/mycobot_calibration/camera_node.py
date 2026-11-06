"""USB camera via OpenCV VideoCapture -> /camera/image_raw + /camera/camera_info.

  ros2 run mycobot_calibration camera --ros-args -p device:=0 \
      -p camera_calibration_file:=$(ros2 pkg prefix mycobot_calibration)/share/mycobot_calibration/config/camera.yaml

camera_info is filled from camera_calibration_file (OpenCV / ROS yaml); without it,
camera_info is not published (the hand-eye node can still read the same file itself).
"""
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from mycobot_calibration.intrinsics import camera_info_msg, load_intrinsics


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera')
        p = self.declare_parameter
        dev = p('device', '0').value                 # index ("0") or path ("/dev/video2")
        w, h = p('width', 640).value, p('height', 480).value
        fps = p('fps', 30.0).value
        self.frame_id = p('frame_id', 'camera_optical_frame').value
        calib = p('camera_calibration_file', '').value
        self.cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open camera {dev}')
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info = None
        if calib:
            K, D, size = load_intrinsics(os.path.expanduser(calib))
            if size is not None and tuple(size) != (aw, ah):
                self.get_logger().warn(f'calibration was made at {size}, camera runs at '
                                       f'{(aw, ah)}: intrinsics do not match!')
            self.info = camera_info_msg(K, D, (aw, ah), self.frame_id)
        self.bridge = CvBridge()
        self.img_pub = self.create_publisher(Image, 'camera/image_raw', 2)
        self.info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 2)
        self.create_timer(1.0 / fps, self._grab)
        self.get_logger().info(f'camera {dev}: {aw}x{ah} @ {fps:.0f} fps, '
                               f'camera_info {"from " + calib if calib else "OFF"}')

    def _grab(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('frame grab failed', throttle_duration_sec=2.0)
            return
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.img_pub.publish(msg)
        if self.info is not None:
            self.info.header = msg.header
            self.info_pub.publish(self.info)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
