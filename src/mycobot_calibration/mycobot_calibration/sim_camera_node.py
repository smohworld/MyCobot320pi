"""Simulated eye-in-hand camera for testing the hand-eye pipeline in RViz.

Renders the ChArUco board as the camera would see it, using
  T_base_cam_true = T_base_link6 (TF, from /joint_states) @ X_true
where X_true = (URDF nominal link6->camera_optical_frame) @ true_offset.
The calibration should recover X_true, not the URDF value.

Publishes: image (mono8), camera_info, static TF link6 -> camera_optical_frame_true,
and the true board as a MarkerArray (ns board_true).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import MarkerArray

from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import (
    fill_transform_msg, inv, transform_msg_to_T, xyzrpy_to_T)
from mycobot_calibration.markers import board_marker
from mycobot_calibration.pose_planner import nominal_board_pose
from mycobot_calibration.util import default_board_file


class SimCamera(Node):

    def __init__(self):
        super().__init__('sim_camera')
        p = self.declare_parameter
        p('board_file', default_board_file())
        p('board_center_xyz', [0.0, -0.22, 0.0])   # board centre in base [m], lying on table
        p('board_yaw', 0.0)
        # X_true = X_urdf @ offset  [x y z roll pitch yaw] (m, rad)
        p('true_offset', [0.004, -0.003, 0.002, 0.035, -0.026, 0.052])
        p('width', 1280)
        p('height', 720)
        p('hfov_deg', 60.0)
        p('distortion', [-0.08, 0.03, 0.0, 0.0, 0.0])
        p('pixel_noise', 3.0)                      # grey-level std
        p('rate', 5.0)
        p('base_frame', 'base')
        p('flange_frame', 'link6')
        p('camera_frame', 'camera_optical_frame')
        p('image_topic', 'camera/image_raw')
        p('camera_info_topic', 'camera/camera_info')
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.target = CharucoTarget(g('board_file'))
        self.w, self.h = int(g('width')), int(g('height'))
        f = self.w / 2.0 / np.tan(np.radians(g('hfov_deg')) / 2.0)
        self.K = np.array([[f, 0, (self.w - 1) / 2.0], [0, f, (self.h - 1) / 2.0], [0, 0, 1.0]])
        self.D = np.array(g('distortion'), float)
        self.noise = float(g('pixel_noise'))
        self.base, self.flange, self.cam = g('base_frame'), g('flange_frame'), g('camera_frame')
        self.offset = xyzrpy_to_T(np.array(g('true_offset'), float))

        T_b = nominal_board_pose(np.zeros(3), float(g('board_yaw')))
        c = T_b[:3, :3] @ self.target.center + T_b[:3, 3]
        T_b[:3, 3] += np.array(g('board_center_xyz'), float) - c
        self.T_base_board = T_b

        self.bridge = CvBridge()
        self.rng = np.random.default_rng()
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.img_pub = self.create_publisher(Image, g('image_topic'), 2)
        self.info_pub = self.create_publisher(CameraInfo, g('camera_info_topic'), 2)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, 'sim_camera/markers', latched)
        self.X_true = None
        self.create_timer(1.0 / float(g('rate')), self._tick)
        self.get_logger().info('sim camera: waiting for TF ...')

    def _lookup(self, parent, child):
        try:
            return transform_msg_to_T(
                self.tf_buf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001 - TF not ready yet
            return None

    def _tick(self):
        if self.X_true is None:
            X_urdf = self._lookup(self.flange, self.cam)
            if X_urdf is None:
                return
            self.X_true = X_urdf @ self.offset
            ts = TransformStamped()
            ts.header.stamp = self.get_clock().now().to_msg()
            ts.header.frame_id = self.flange
            ts.child_frame_id = self.cam + '_true'
            fill_transform_msg(ts.transform, self.X_true)
            self.static_tf.sendTransform(ts)
            ma = MarkerArray()
            ma.markers = board_marker(self.target, self.T_base_board, self.base, 'board_true',
                                      alpha=1.0, stamp=ts.header.stamp)
            self.marker_pub.publish(ma)
            self.get_logger().info('sim camera running (ground truth TF: '
                                   f'{self.flange} -> {self.cam}_true)')
        T_bl6 = self._lookup(self.base, self.flange)
        if T_bl6 is None:
            return
        T_cam_board = inv(T_bl6 @ self.X_true) @ self.T_base_board
        img = self.target.render(T_cam_board, self.K, self.D, self.w, self.h)
        if self.noise > 0:
            img = np.clip(img + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        stamp = self.get_clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(img, 'mono8')
        msg.header.stamp, msg.header.frame_id = stamp, self.cam
        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = self.w, self.h
        info.distortion_model = 'plumb_bob'
        info.d = list(map(float, self.D))
        info.k = list(map(float, self.K.ravel()))
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        P = np.hstack([self.K, np.zeros((3, 1))])
        info.p = list(map(float, P.ravel()))
        self.img_pub.publish(msg)
        self.info_pub.publish(info)


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
