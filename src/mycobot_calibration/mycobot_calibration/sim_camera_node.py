"""Synthetic flange camera for testing the hand-eye pipeline without hardware.

Renders the ChArUco board lying on the table as seen from
  T_base_cam_true = T_base_link6 (TF) @ X_nominal (URDF) @ true_offset
and publishes /camera/image_raw + /camera/camera_info. TF link6 -> camera_optical_frame_true
holds the ground truth, so the calibrator can report its error.
"""
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import (
    fill_transform_msg, inv, transform_msg_to_T, xyzrpy_to_T)
from mycobot_calibration.intrinsics import camera_info_msg
from mycobot_calibration.pose_generator import board_pose_flat


class SimCamera(Node):

    def __init__(self):
        super().__init__('sim_camera')
        p = self.declare_parameter
        board_file = p('board_file', '').value
        self.target = CharucoTarget(board_file)
        self.T_board = board_pose_flat(self.target, p('board_center_xyz', [0.22, 0.0, 0.0]).value,
                                       p('board_yaw', 0.0).value)
        self.offset = xyzrpy_to_T(np.array(
            p('true_offset_xyzrpy', [0.004, -0.003, 0.005, 0.02, -0.015, 0.03]).value, float))
        self.w, self.h = p('width', 640).value, p('height', 480).value
        f = p('focal_px', 600.0).value
        self.noise = p('noise_sigma', 2.0).value
        self.K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1.0]])
        self.flange = p('flange_frame', 'link6').value
        self.cam = p('camera_frame', 'camera_optical_frame').value
        self.info = camera_info_msg(self.K, np.zeros(5), (self.w, self.h), self.cam)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.static = StaticTransformBroadcaster(self)
        self.bridge = CvBridge()
        self.X_true = None
        self.rng = np.random.default_rng(1)
        self.img_pub = self.create_publisher(Image, 'camera/image_raw', 2)
        self.info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 2)
        self.create_timer(0.1, self._tick)

    def _T(self, a, b):
        try:
            return transform_msg_to_T(self.tf.lookup_transform(a, b, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def _tick(self):
        if self.X_true is None:
            X_nom = self._T(self.flange, self.cam)
            if X_nom is None:
                return
            self.X_true = X_nom @ self.offset
            ts = TransformStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = self.flange
            ts.child_frame_id = self.cam + '_true'
            fill_transform_msg(ts.transform, self.X_true)
            self.static.sendTransform(ts)
            self.get_logger().info('sim camera running (ground truth TF '
                                   f'{self.flange} -> {self.cam}_true)')
        T_bl6 = self._T('base', self.flange)
        if T_bl6 is None:
            return
        img = self.target.render(inv(T_bl6 @ self.X_true) @ self.T_board, self.K, self.w, self.h)
        if self.noise > 0:
            img = np.clip(img + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        msg = self.bridge.cv2_to_imgmsg(img, 'mono8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.cam
        self.img_pub.publish(msg)
        self.info.header = msg.header
        self.info_pub.publish(self.info)


def main(args=None):
    rclpy.init(args=args)
    node = SimCamera()
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
