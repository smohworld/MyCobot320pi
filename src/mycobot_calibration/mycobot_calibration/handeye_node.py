"""Automatic eye-in-hand calibration driven by MoveIt2 (myCobot 320 + flange camera).

Services (std_srvs/Trigger) under /handeye_calibrator/:
  plan     find the board, generate viewpoints, solve IK (MoveIt /compute_ik, collision
           checked, joint limits incl. the J6 +-90 deg cable window) - no motion
  run      plan if needed, then for every pose: MoveIt plan+execute -> settle -> capture;
           finally solve and save
  capture  one sample at the current pose (manual mode: move with RViz / by hand)
  solve    solve with the samples so far
  clear    forget plan and samples
  abort    stop `run` (cancels the current motion)
  verify_views / verify_touch   check camera -> base after calibration (verify.py)

Intrinsics: camera_calibration_file (OpenCV / ROS yaml from your camera calibration)
or, if empty, camera_info_topic.

Output (output_dir/<date_time>/): images/, samples.yaml, result.yaml, planned_poses.yaml
and hand_eye.yaml -> ros2 launch mycobot_moveit_config bringup.launch.py calibration_file:=...
RViz: ~/markers (board, planned cameras), ~/debug_image, TF link6 -> camera_optical_frame_calib
"""
import csv
import datetime
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from mycobot_calibration import solver
from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import (
    T_to_xyzrpy, fill_pose_msg, fill_transform_msg, inv, rot_angle, transform_msg_to_T)
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_calibration.verify import VerifyMixin
from mycobot_calibration.pose_generator import (
    ViewpointGenerator, board_pose_flat, camera_roll, order_by_joint_distance,
    rotation_spread_deg, select_diverse)

ARM_JOINTS = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
              'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']
STATUS_COLOR = {'planned': (0.6, 0.6, 0.6), 'current': (1.0, 0.8, 0.0),
                'ok': (0.1, 0.9, 0.2), 'failed': (0.9, 0.1, 0.1)}


def _Tl(T):
    return [[float(v) for v in row] for row in T]


class HandEyeCalibrator(VerifyMixin, Node):

    def __init__(self):
        super().__init__('handeye_calibrator')
        share = get_package_share_directory('mycobot_calibration')
        desc = get_package_share_directory('mycobot_description')
        p = self.declare_parameter
        p('board_file', os.path.join(share, 'config', 'charuco_board.yaml'))
        p('camera_calibration_file', '')
        p('image_topic', '/camera/image_raw')
        p('camera_info_topic', '/camera/camera_info')
        p('base_frame', 'base')
        p('flange_frame', 'link6')
        p('camera_frame', 'camera_optical_frame')
        p('ik_link', 'gripper_tcp')                   # tip link of the MoveIt 'arm' group
        p('group', 'arm')
        # viewpoints
        p('n_poses', 20)
        p('distance_min', 0.24)
        p('distance_max', 0.36)
        p('tilt_min_deg', 10.0)
        p('tilt_max_deg', 35.0)
        p('roll_range_deg', 30.0)
        p('aim_jitter', 0.25)
        p('max_incidence_deg', 50.0)
        p('image_margin_px', 25)
        p('min_visible_fraction', 0.8)                # of the chessboard corners
        p('min_camera_height', 0.08)                  # camera above table [m]
        p('seed', 0)
        p('enough_feasible', 45)                      # reachable viewpoints to choose from
        p('max_candidates', 200)                      # viewpoints tried at most
        p('ik_timeout', 0.1)                          # per /compute_ik call [s]
        # keep the arm in the start configuration: reject IK solutions farther than this
        # from the start pose, per joint J1..J6 [deg] (<= 0: no limit). Avoids elbow/wrist
        # flips and big J1 swings between viewpoints (cable wrap, wide sweeps).
        p('max_joint_dev_deg', [60.0, 90.0, 70.0, 90.0, 90.0, 90.0])
        p('board_guess_xyz', [0.22, 0.0, -0.007])         # used when the board is not visible
        p('board_guess_yaw', 0.0)
        # MoveIt
        p('add_table', True)
        p('table_z', -0.007)                          # table top in base [m]
        # planning keeps the robot this far above the desk (the arm is ~3-6 mm accurate)
        p('table_margin', 0.012)                      # [m]
        p('table_size_xy', [1.5, 1.5])                # [m]
        p('table_center_xy', [0.0, 0.0])              # box centre in base [m]
        p('velocity_scaling', 0.3)
        p('accel_scaling', 0.3)
        p('planning_time', 5.0)
        # capture / solve
        p('settle_time', 1.0)
        # joint_states must be this close to the planned pose before capturing [deg]. Keep it
        # above the driver's goal_tolerance_deg (2.0): real servos stop ~1-2 deg off target.
        # The sample uses the MEASURED joints (TF), so this check is only a sanity check.
        p('pose_tolerance_deg', 6.0)
        # joint_states older than this count as missing (driver stops publishing when it
        # cannot read the joints; a frozen last value must not be used for a sample)
        p('joint_state_timeout', 1.5)                 # [s] > driver publish gap on a slow link
        p('min_corners', 12)
        p('max_pnp_rms_px', 1.0)
        p('poses_file', '')                           # replay planned_poses.yaml
        p('return_to_start', True)
        p('auto_start', False)
        p('output_dir', os.path.expanduser('~/handeye_calib'))
        p('base_calibration_file', os.path.join(desc, 'config', 'hand_eye_nominal.yaml'))
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        g = self.g

        self.target = CharucoTarget(os.path.expanduser(g('board_file')))
        self.get_logger().info(f"board: {self.target.describe()} ({g('board_file')})")
        self.base, self.flange, self.cam = g('base_frame'), g('flange_frame'), g('camera_frame')
        self.K = self.D = self.img_size = None
        if g('camera_calibration_file'):
            self.K, self.D, self.img_size = load_intrinsics(g('camera_calibration_file'))
            self.get_logger().info(f"intrinsics from {g('camera_calibration_file')}: "
                                   f"fx {self.K[0, 0]:.1f} fy {self.K[1, 1]:.1f}")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = None                            # (stamp_sec, image)
        self.js = {}
        self.js_t = None                              # monotonic time of the last joint_states
        self.plan = []                                # list of (T_base_cam, q)
        self.plan_status = []
        self.samples = []
        self.result = None
        self.T_board = None
        self.start_q = None
        self.run_dir = None
        self._abort = threading.Event()
        self._worker = None
        self._last_preview = 0.0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sens = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, g('image_topic'), self._on_image, 2, callback_group=sens)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._on_info, 2,
                                 callback_group=sens)
        self.create_subscription(JointState, 'joint_states', self._on_js, 10, callback_group=sens)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', latched)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 2)
        self.status_pub = self.create_publisher(String, '~/status', latched)
        self.moveit = MoveItClient(self, g('group'), self.base)

        cb = ReentrantCallbackGroup()
        for name, fn in [('plan', self._srv_plan), ('run', self._srv_run),
                         ('capture', self._srv_capture), ('solve', self._srv_solve),
                         ('clear', self._srv_clear), ('abort', self._srv_abort)]:
            self.create_service(Trigger, '~/' + name, fn, callback_group=cb)
        self._arm_joints = ARM_JOINTS
        self._init_verify(cb)
        if g('auto_start'):
            self._auto = self.create_timer(3.0, self._auto_start, callback_group=cb)
        self.status('ready: call ~/plan (preview), ~/run (auto) or ~/capture (manual)')

    # ------------------------------------------------------------- inputs
    def _on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.latest = (stamp, img)
        now = time.monotonic()
        busy = self._worker is not None and self._worker.is_alive()
        if self.K is not None and not busy and now - self._last_preview > 0.5:
            self._last_preview = now
            self._publish_debug(img, self.target.detect(img, self.K, self.D,
                                                        self.g('min_corners')), msg.header)

    def _on_info(self, msg):
        if self.g('camera_calibration_file'):
            return
        if self.K is None:
            self.get_logger().info(f'intrinsics from camera_info ({msg.width}x{msg.height})')
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.D = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
        self.img_size = (msg.width, msg.height)

    def _on_js(self, msg):
        with self.lock:
            self.js.update(zip(msg.name, msg.position))
            if all(n in msg.name for n in ARM_JOINTS):
                self.js_t = time.monotonic()

    # ------------------------------------------------------------ helpers
    def status(self, text, level='info'):
        log = self.get_logger()
        if level == 'error':
            log.error(text)
        elif level == 'warn':
            log.warn(text)
        else:
            log.info(text)
        self.status_pub.publish(String(data=text))

    def lookup(self, parent, child):
        try:
            return transform_msg_to_T(
                self.tf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def js_age(self):
        with self.lock:
            return float('inf') if self.js_t is None else time.monotonic() - self.js_t

    def current_q(self):
        """Arm joints [rad], or None when joint_states are missing or stale."""
        if self.js_age() > float(self.g('joint_state_timeout')):
            return None
        with self.lock:
            if not all(n in self.js for n in ARM_JOINTS):
                return None
            return np.array([self.js[n] for n in ARM_JOINTS])

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def fresh_image(self, after, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                li = self.latest
            if li is not None and li[0] >= after:
                return li[1]
            time.sleep(0.03)
        return None

    def wait_ready(self, timeout=15.0, need_moveit=True):
        t0 = time.monotonic()
        missing = []
        while time.monotonic() - t0 < timeout:
            missing = []
            if self.K is None:
                missing.append('intrinsics (camera_calibration_file or camera_info)')
            if self.latest is None:
                missing.append(f"image on {self.g('image_topic')}")
            if self.current_q() is None:
                missing.append('joint_states')
            if self.lookup(self.flange, self.cam) is None:
                missing.append(f'TF {self.flange}->{self.cam}')
            if need_moveit and not self.moveit.wait_ready(timeout=0.5):
                missing.append('move_group (/compute_ik, /move_action)')
            if not missing:
                if self.img_size is None:
                    self.img_size = self.latest[1].shape[::-1]
                return True
            time.sleep(0.3)
        self.status(f'not ready, missing: {missing}', 'error')
        return False

    def _publish_debug(self, img, det, header=None):
        msg = self.bridge.cv2_to_imgmsg(self.target.draw(img, det, self.K, self.D), 'bgr8')
        if header is not None:
            msg.header = header
        self.debug_pub.publish(msg)

    # ------------------------------------------------------------ markers
    def publish_markers(self):
        ma = MarkerArray()
        m = Marker(action=Marker.DELETEALL)
        m.header.frame_id = self.base
        ma.markers.append(m)
        boards = [('board_estimate', self.T_board, (0.2, 0.4, 1.0, 0.4))]
        if self.result is not None:
            boards.append(('board_calibrated', self.result['T_base_board'], (0.1, 0.9, 0.2, 0.5)))
        for ns, T, rgba in boards:
            if T is None:
                continue
            m = Marker(ns=ns, id=0, type=Marker.CUBE, action=Marker.ADD)
            m.header.frame_id = self.base
            Tc = T.copy()
            Tc[:3, 3] = T[:3, :3] @ self.target.center + T[:3, 3]
            fill_pose_msg(m.pose, Tc)
            m.scale.x, m.scale.y, m.scale.z = self.target.size[0], self.target.size[1], 0.002
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            ma.markers.append(m)
        for i, ((T, _), st) in enumerate(zip(self.plan, self.plan_status)):
            m = Marker(ns='cameras', id=i, type=Marker.ARROW, action=Marker.ADD)
            m.header.frame_id = self.base
            p0 = T[:3, 3]
            p1 = p0 + 0.06 * T[:3, 2]
            m.points = [Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])),
                        Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))]
            m.scale.x, m.scale.y, m.scale.z = 0.006, 0.012, 0.015
            m.color.r, m.color.g, m.color.b = STATUS_COLOR[st]
            m.color.a = 1.0
            ma.markers.append(m)
            t = Marker(ns='labels', id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.header.frame_id = self.base
            t.pose.position.x, t.pose.position.y, t.pose.position.z = \
                float(p0[0]), float(p0[1]), float(p0[2]) + 0.015
            t.scale.z = 0.015
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = str(i + 1)
            ma.markers.append(t)
        self.marker_pub.publish(ma)

    # ----------------------------------------------------------- planning
    def estimate_board(self):
        X = self.lookup(self.flange, self.cam)
        T_bl6 = self.lookup(self.base, self.flange)
        img = self.fresh_image(self.now() - 0.5)
        if img is not None and T_bl6 is not None:
            det = self.target.detect(img, self.K, self.D, self.g('min_corners'))
            self._publish_debug(img, det)
            if det['ok']:
                self.T_board = T_bl6 @ X @ det['T_cam_board']
                c = self.T_board[:3, :3] @ self.target.center + self.T_board[:3, 3]
                self.status(f'board found at {np.round(c, 3).tolist()} (via nominal hand-eye)')
                return
        self.T_board = board_pose_flat(self.target, self.g('board_guess_xyz'),
                                       self.g('board_guess_yaw'))
        self.status('board not visible: using board_guess_xyz / board_guess_yaw', 'warn')

    def add_table(self):
        """Table box in the planning scene, its top raised by table_margin: the arm is only
        ~3-6 mm accurate, so MoveIt must keep that much (and more) away from the real desk."""
        g = self.g
        if not g('add_table'):
            return
        z = float(g('table_z')) + float(g('table_margin'))
        cx, cy = (float(v) for v in g('table_center_xy'))
        sx, sy = (float(v) for v in g('table_size_xy'))
        ok = self.moveit.add_box('calibration_table', [cx, cy, z - 0.025], [sx, sy, 0.05])
        self.status(f"table added to the planning scene (desk z={g('table_z'):.3f} m, "
                    f"collision top z={z:.3f} m incl. table_margin)" if ok else
                    'could not add the table to the planning scene', 'info' if ok else 'warn')

    def make_plan(self, seed=None, n_poses=None, use_poses_file=True):
        if not self.wait_ready():
            return False
        g = self.g
        self.add_table()
        q0 = self.current_q()
        self.start_q = q0
        X = self.lookup(self.flange, self.cam)
        T_l6_ik = self.lookup(self.flange, g('ik_link'))
        if T_l6_ik is None:
            self.status(f"no TF {self.flange}->{g('ik_link')}", 'error')
            return False

        n_poses = int(g('n_poses') if n_poses is None else n_poses)
        seed = int(g('seed') if seed is None else seed)
        if use_poses_file and g('poses_file'):
            with open(os.path.expanduser(g('poses_file'))) as f:
                d = yaml.safe_load(f)
            T_bl6_now = self.lookup(self.base, self.flange)
            self.plan = [(None, np.array(q, float)) for q in d['joint_positions']]
            # camera poses only for display: unknown without FK, show the current one
            self.plan = [(T_bl6_now @ X, q) for _, q in self.plan]
            self.status(f"loaded {len(self.plan)} poses from {g('poses_file')}")
        else:
            self.estimate_board()
            gen = ViewpointGenerator(self.target, n_poses,
                                     (g('distance_min'), g('distance_max')),
                                     (g('tilt_min_deg'), g('tilt_max_deg')),
                                     g('roll_range_deg'), g('aim_jitter'), seed)
            T_cam_now = self.lookup(self.base, self.flange) @ X
            ref = gen.reference_frame(self.T_board, T_cam_now[:3, 3])
            ref_roll = camera_roll(T_cam_now, ref)
            n_want = n_poses
            enough = max(n_want, int(g('enough_feasible')))
            self.status(f'planning: collecting up to {enough} reachable viewpoints '
                        f'(MoveIt IK, collision checked), then picking the {n_want} most '
                        'different ones ...')
            feasible, tried, t0 = [], 0, time.monotonic()
            dev = np.radians(np.asarray(g('max_joint_dev_deg'), float))
            dev = np.where(dev > 0, dev, np.inf)
            far = 0                                   # IK solutions outside the window
            for i, cands in gen.slots(self.T_board, ref_roll, max_slots=int(g('max_candidates'))):
                if self._abort.is_set() or len(feasible) >= enough:
                    break
                tried += 1
                for T_cam in cands:
                    if T_cam[2, 3] < g('table_z') + g('min_camera_height'):
                        continue
                    if not self.target.visible(inv(T_cam) @ self.T_board, self.K, self.D,
                                               *self.img_size, g('image_margin_px'),
                                               g('max_incidence_deg'),
                                               g('min_visible_fraction')):
                        continue
                    T_ik = T_cam @ inv(X) @ T_l6_ik
                    q = self.moveit.compute_ik(T_ik, g('ik_link'), ARM_JOINTS, q0,
                                               timeout=float(g('ik_timeout')))
                    if q is None:
                        continue
                    if np.any(np.abs(np.asarray(q) - np.asarray(q0)) > dev):
                        far += 1                      # reachable, but in another arm config
                        continue
                    feasible.append((T_cam, np.array(q)))
                    break
                if tried % 20 == 0:
                    self.get_logger().info(f'  {tried} viewpoints tried, {len(feasible)} reachable')
            self.get_logger().info(f'{len(feasible)} reachable of {tried} viewpoints tried '
                                   f'({time.monotonic() - t0:.0f} s), {far} IK solutions '
                                   'rejected by max_joint_dev_deg')
            plan = select_diverse(feasible, n_want, q0)
            if len(plan) < min(3, n_want):        # verify_touch asks for a single view
                self.status(f'only {len(plan)} reachable viewpoints: move the board closer to '
                            'the robot or adjust distance_*/tilt_* / board_guess_*'
                            + (f' ({far} rejected by max_joint_dev_deg: start with the camera '
                               'looking at the board, or widen it)' if far else ''), 'error')
                self.plan, self.plan_status = plan, ['failed'] * len(plan)
                self.publish_markers()
                return False
            self.plan = order_by_joint_distance(q0, plan)
            Tg = [T @ inv(X) for T, _ in self.plan]
            mx, med = rotation_spread_deg(Tg)
            self.status(f'planned {len(self.plan)}/{n_poses} poses '
                        f'(chosen from {len(feasible)} reachable, {tried} tried), flange rotation '
                        f'spread max {mx:.0f} / median {med:.0f} deg')
            if mx < 30:
                self.status('little rotation between poses: raise tilt_max_deg / '
                            'roll_range_deg for a better calibration', 'warn')
        self.plan_status = ['planned'] * len(self.plan)
        self.publish_markers()
        return True

    # ------------------------------------------------------------ capture
    def _log_capture(self, idx, ok, why, det, img, q, jit):
        """One row per capture attempt (also rejected ones) in <run_dir>/capture_log.csv:
        what the arm really did (planned vs read joints, shake while waiting) and how good the
        detection was, so a bad sample can be explained afterwards."""
        try:
            self._ensure_run_dir()
            f = os.path.join(self.run_dir, 'capture_log.csv')
            new = not os.path.exists(f)
            qp = (np.asarray(self.plan[idx][1], float)
                  if idx is not None and idx < len(self.plan) else None)
            qr = q if q is not None else np.full(6, np.nan)
            sharp = bright = float('nan')
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                bright = float(gray.mean())
            with open(f, 'a', newline='') as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(['time', 'pose_idx', 'ok', 'reason', 'n_corners', 'pnp_rms_px',
                                'sharpness', 'brightness', 'js_age_s']
                               + [f'{k}{j}' for k in ('q_read_deg', 'q_err_deg', 'js_std_deg',
                                                      'js_p2p_deg') for j in range(1, 7)])
                err = np.degrees(qp - qr) if qp is not None else np.full(6, np.nan)
                w.writerow([f'{time.time():.2f}', '' if idx is None else idx, int(ok), why,
                            det['n'] if det else '', f"{det['rms']:.3f}" if det else '',
                            f'{sharp:.1f}', f'{bright:.1f}', f'{self.js_age():.3f}']
                           + [f'{v:.3f}' for v in np.concatenate(
                               [np.degrees(qr), err, jit[0], jit[1]])])
        except Exception as e:  # noqa: BLE001 - logging must never break a capture
            self.get_logger().warn(f'capture_log write failed: {e!r}')

    def _watch_joints(self, dur=0.5):
        """Joint readings over `dur` s: std and peak-to-peak [deg] (shake after arrival)."""
        qs, t0 = [], time.monotonic()
        while time.monotonic() - t0 < dur:
            q = self.current_q()
            if q is not None:
                qs.append(q)
            time.sleep(0.05)
        if len(qs) < 3:
            return np.full(6, np.nan), np.full(6, np.nan)
        qs = np.degrees(np.array(qs))
        return qs.std(axis=0), np.ptp(qs, axis=0)

    def capture_here(self, idx=None):
        time.sleep(float(self.g('settle_time')))
        jit = self._watch_joints()
        img = self.fresh_image(self.now())
        T_bl6 = self.lookup(self.base, self.flange)
        q = self.current_q()
        if img is None or T_bl6 is None or q is None:
            self.status('no fresh image / TF' if q is not None else
                        f'joint_states stale ({self.js_age():.1f} s): sample rejected', 'error')
            self._log_capture(idx, False, 'no image/TF' if q is not None else 'js stale',
                              None, img, q, jit)
            return False
        det = self.target.detect(img, self.K, self.D, self.g('min_corners'))
        self._publish_debug(img, det)
        if not det['ok'] or det['rms'] > self.g('max_pnp_rms_px'):
            why = f"{det['n']} corners" if not det['ok'] else f"PnP rms {det['rms']:.2f}px"
            self.status(f'sample rejected ({why})', 'warn')
            self._log_capture(idx, False, why, det, img, q, jit)
            return False
        k = len(self.samples)
        self.samples.append({
            'T_base_link6': T_bl6, 'T_cam_board': det['T_cam_board'],
            'ids': det['ids'].reshape(-1), 'pixels': det['corners'].reshape(-1, 2).astype(float),
            'K': self.K.copy(), 'D': self.D.copy(), 'q': q, 'pose_index': idx,
            'pnp_rms': det['rms']})
        self._ensure_run_dir()
        cv2.imwrite(os.path.join(self.run_dir, 'images', f'{k:02d}.png'), img)
        self._log_capture(idx, True, f'sample {k}', det, img, q, jit)
        self.status(f"sample {k + 1}: {det['n']}/{self.target.n_corners} corners, "
                    f"PnP rms {det['rms']:.2f}px")
        return True

    def _ensure_run_dir(self):
        if self.run_dir is None:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = os.path.join(os.path.expanduser(self.g('output_dir')), stamp)
            os.makedirs(os.path.join(self.run_dir, 'images'), exist_ok=True)
            self.get_logger().info(f'saving to {self.run_dir}')

    # -------------------------------------------------------------- solve
    def solve(self):
        if len(self.samples) < 3:
            self.status(f'need >= 3 samples (have {len(self.samples)}), 15+ recommended', 'error')
            return False
        try:
            r = solver.calibrate(self.samples, self.target.corners3d)
        except Exception as e:  # noqa: BLE001
            self.status(f'solve failed: {e}', 'error')
            return False
        self.result = r
        X = r['X']
        v = T_to_xyzrpy(X)
        mx, med = solver.motion_diversity([s['T_base_link6'] for s in self.samples])
        lines = [f"hand-eye result: {r['n_samples']} samples, method {r['method']}",
                 f"  link6 -> {self.cam}: xyz [m] {np.round(v[:3], 5).tolist()} "
                 f"rpy [rad] {np.round(v[3:], 5).tolist()}",
                 f"  base->board scatter {r['board_trans_rms_mm']:.2f} mm / "
                 f"{r['board_rot_rms_deg']:.3f} deg rms, reprojection {r['chain_reproj_rms_px']:.2f} px",
                 f"  flange rotation spread max {mx:.0f} / median {med:.0f} deg"]
        for name, c in r['candidates'].items():
            e = inv(X) @ c['X']
            lines.append(f"    {name:10s} scatter {c['board_trans_rms_mm']:6.2f} mm, diff "
                         f"{np.linalg.norm(e[:3, 3]) * 1000:5.2f} mm "
                         f"{np.degrees(rot_angle(e[:3, :3])):5.2f} deg")
        X_nom = self.lookup(self.flange, self.cam)
        if X_nom is not None:
            e = inv(X_nom) @ X
            lines.append(f"  vs current URDF: {np.linalg.norm(e[:3, 3]) * 1000:.1f} mm, "
                         f"{np.degrees(rot_angle(e[:3, :3])):.2f} deg")
        X_true = self.lookup(self.flange, self.cam + '_true')
        if X_true is not None:
            e = inv(X_true) @ X
            lines.append(f"  vs SIM GROUND TRUTH: {np.linalg.norm(e[:3, 3]) * 1000:.2f} mm, "
                         f"{np.degrees(rot_angle(e[:3, :3])):.3f} deg")
        for ln in lines:
            self.get_logger().info(ln)
        if r['board_trans_rms_mm'] > 3.0:
            self.status('scatter > 3 mm: check board measurements (square_length), '
                        'intrinsics and joint directions/offsets', 'warn')

        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.flange
        ts.child_frame_id = self.cam + '_calib'
        fill_transform_msg(ts.transform, X)
        self.static_tf.sendTransform(ts)
        self.publish_markers()
        self._save(r, v, lines)
        return True

    def _save(self, r, v, lines):
        self._ensure_run_dir()
        d = self.run_dir
        with open(os.path.join(d, 'samples.yaml'), 'w') as f:
            yaml.safe_dump({'board_file': self.g('board_file'),
                            'K': _Tl(self.samples[0]['K']),
                            'D': [float(x) for x in self.samples[0]['D']],
                            'samples': [{
                                'q': None if s['q'] is None else [float(x) for x in s['q']],
                                'T_base_link6': _Tl(s['T_base_link6']),
                                'T_cam_board': _Tl(s['T_cam_board']),
                                'ids': [int(x) for x in s['ids']],
                                'pixels': [[float(a), float(b)] for a, b in s['pixels']],
                                'pnp_rms_px': float(s['pnp_rms'])} for s in self.samples]},
                           f, sort_keys=False)
        with open(os.path.join(d, 'result.yaml'), 'w') as f:
            yaml.safe_dump({
                'method': r['method'], 'n_samples': r['n_samples'],
                'link6_to_camera_optical_T': _Tl(r['X']), 'T_base_board': _Tl(r['T_base_board']),
                'board_trans_rms_mm': r['board_trans_rms_mm'],
                'board_rot_rms_deg': r['board_rot_rms_deg'],
                'board_trans_per_sample_mm': r['board_trans_per_sample_mm'],
                'chain_reproj_rms_px': r['chain_reproj_rms_px'],
                'chain_reproj_per_sample_px': r['chain_reproj_per_sample_px'],
                'candidates': {k: {'board_trans_rms_mm': float(c['board_trans_rms_mm']),
                                   'board_rot_rms_deg': float(c['board_rot_rms_deg']),
                                   'T': _Tl(c['X'])} for k, c in r['candidates'].items()},
                'log': lines}, f, sort_keys=False)
        try:
            with open(os.path.expanduser(self.g('base_calibration_file'))) as f:
                base = yaml.safe_load(f) or {}
        except OSError:
            base = {}
        base['link6_to_camera_optical'] = dict(zip(
            ['x', 'y', 'z', 'roll', 'pitch', 'yaw'], [round(float(x), 6) for x in v]))
        # The TCP is not calibrated here: keep the one of the URDF that is running (TF), not the
        # nominal 0.165 of base_calibration_file.
        T_tcp = self.lookup(self.flange, 'gripper_tcp')
        if T_tcp is not None:
            base['link6_to_gripper_tcp'] = dict(zip(
                ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
                [round(float(x), 6) for x in T_to_xyzrpy(T_tcp)]))
        with open(os.path.join(d, 'hand_eye.yaml'), 'w') as f:
            f.write(f"# hand-eye calibration {os.path.basename(d)}: {r['method']}, "
                    f"{r['n_samples']} samples, board scatter {r['board_trans_rms_mm']:.2f} mm\n")
            yaml.safe_dump(base, f, sort_keys=False)
        try:      # per-sample diagnostics: which poses carry the error, and why
            with open(os.path.join(d, 'per_sample.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['sample', 'pose_idx', 'board_err_mm', 'reproj_px', 'pnp_rms_px',
                            'n_corners', 'board_tilt_deg', 'cam_dist_m']
                           + [f'q{j}_deg' for j in range(1, 7)])
                for i, smp in enumerate(self.samples):
                    Tcb = smp['T_cam_board']
                    tilt = np.degrees(np.arccos(np.clip(abs(Tcb[2, 2]), 0, 1)))
                    q = np.degrees(smp['q']) if smp['q'] is not None else np.full(6, np.nan)
                    w.writerow([i, smp.get('pose_index', ''),
                                f"{r['board_trans_per_sample_mm'][i]:.3f}",
                                f"{r['chain_reproj_per_sample_px'][i]:.3f}",
                                f"{smp['pnp_rms']:.3f}", len(smp['ids']), f'{tilt:.1f}',
                                f'{np.linalg.norm(Tcb[:3, 3]):.3f}']
                               + [f'{v:.2f}' for v in q])
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'per_sample.csv failed: {e!r}')
        if self.plan:
            with open(os.path.join(d, 'planned_poses.yaml'), 'w') as f:
                yaml.safe_dump({'joint_names': ARM_JOINTS, 'joint_positions': [
                    [float(x) for x in q] for _, q in self.plan]}, f)
        self.status(f'saved {d}/hand_eye.yaml -> calibration_file:={d}/hand_eye.yaml')

    # ---------------------------------------------------------------- run
    def _move(self, q):
        g = self.g
        return self.moveit.move_joints(ARM_JOINTS, q, g('velocity_scaling'), g('accel_scaling'),
                                       g('planning_time'))

    def _at_pose(self, q, i, n, timeout=3.0):
        """joint_states must agree with the planned pose before an image is used
        (catches a stale / second /joint_states publisher or a motion that stopped early)."""
        tol_deg = float(self.g('pose_tolerance_deg'))
        t0 = time.monotonic()
        err = float('inf')
        while time.monotonic() - t0 < timeout:
            q_now = self.current_q()
            if q_now is not None:
                err = float(np.degrees(np.max(np.abs(q_now - np.asarray(q)))))
                if err <= tol_deg:
                    return True
            time.sleep(0.1)
        self.status(f'pose {i + 1}/{n}: joint_states {err:.1f} deg away from the planned pose '
                    f'(> pose_tolerance_deg {tol_deg:.1f}), skipped (servo not settled, or '
                    'another node publishing /joint_states?)', 'warn')
        return False

    def _retry_capture(self, i):
        self.status('retrying capture once')
        return self.capture_here(i)

    def _run(self):
        self._abort.clear()
        if not self.plan and not self.make_plan():
            return
        self.samples, self.result, self.run_dir = [], None, None
        n = len(self.plan)
        for i, (_, q) in enumerate(self.plan):
            if self._abort.is_set():
                self.status('aborted', 'warn')
                break
            self.plan_status[i] = 'current'
            self.publish_markers()
            self.status(f'pose {i + 1}/{n}: moving')
            ok, msg = self._move(q)
            if not ok:
                # the driver aborts (CONTROL_FAILED) when a servo stops just outside its
                # goal_tolerance_deg; the sample uses the measured joints, so it is still
                # usable if the arm is within pose_tolerance_deg of the planned pose
                self.status(f'pose {i + 1}/{n}: motion failed ({msg}), checking where the '
                            'arm stopped', 'warn')
            if not self._abort.is_set():
                ok = self._at_pose(q, i, n) and (self.capture_here(i) or self._retry_capture(i))
            self.plan_status[i] = 'ok' if ok else 'failed'
            self.publish_markers()
        if self.g('return_to_start') and self.start_q is not None and not self._abort.is_set():
            self.status('returning to the start pose')
            ok, msg = self._move(self.start_q)
            if not ok:
                self.status(f'return to start failed ({msg})', 'warn')
        self.status(f'{len(self.samples)}/{n} samples captured')
        self.solve()

    def _start(self, fn):
        if self._worker is not None and self._worker.is_alive():
            return False
        self._abort.clear()
        self._worker = threading.Thread(target=fn, daemon=True)
        self._worker.start()
        return True

    def _auto_start(self):
        self._auto.cancel()
        self._start(self._run)

    # ----------------------------------------------------------- services
    def _srv_plan(self, req, res):
        self.plan, self.plan_status = [], []
        res.success = self._start(self.make_plan)
        res.message = 'planning (see ~/status, RViz)' if res.success else 'busy'
        return res

    def _srv_run(self, req, res):
        res.success = self._start(self._run)
        res.message = 'running (see ~/status, RViz)' if res.success else 'busy'
        return res

    def _srv_capture(self, req, res):
        if self._worker is not None and self._worker.is_alive():
            res.success, res.message = False, 'busy'
            return res
        if not self.wait_ready(timeout=3.0, need_moveit=False):
            res.success, res.message = False, 'not ready (see log)'
            return res
        res.success = self.capture_here()
        res.message = f'{len(self.samples)} samples'
        return res

    def _srv_solve(self, req, res):
        res.success = self.solve()
        res.message = (f"scatter {self.result['board_trans_rms_mm']:.2f} mm"
                       if res.success else 'failed (see log)')
        return res

    def _srv_clear(self, req, res):
        self.samples, self.plan, self.plan_status = [], [], []
        self.result, self.run_dir = None, None
        self.publish_markers()
        res.success, res.message = True, 'cleared'
        return res

    def _srv_abort(self, req, res):
        self._abort.set()
        self.moveit.cancel()
        res.success, res.message = True, 'abort requested'
        return res


def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibrator()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
