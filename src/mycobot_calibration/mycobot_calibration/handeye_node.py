"""Automatic eye-in-hand calibration (myCobot 320 + camera flange), RViz-friendly.

Services (std_srvs/Trigger), all under ~/ (e.g. /handeye_calibrator/run):
  plan     estimate the board, generate + show the calibration poses (no motion)
  run      plan (if needed), move through every pose, capture, then solve
  capture  capture one sample at the current pose (manual / hand-teach mode)
  solve    solve with the samples collected so far
  clear    forget samples and plan
  abort    stop a running `run` after the current pose

Pose source: auto-generated around the board (default) or `poses_file`
(yaml with `joint_positions: [[q1..q6], ...]` in rad, e.g. a previous run's
planned_poses.yaml or poses taught by hand).

RViz: ~/markers (board estimate, planned camera frustums coloured by status,
calibrated board), ~/debug_image (detections), TF link6 -> camera_optical_frame_calib.

Method: per-sample ChArUco PnP, then AX = XB (OpenCV Tsai / Park / Horaud /
Andreff / Daniilidis) and a reprojection refinement; the final X is the candidate
with the smallest base->board scatter. Output folder contains the images,
samples, report and a hand_eye.yaml for mycobot_description (calibration_file:=).
"""
import datetime
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import MarkerArray

from mycobot_calibration import solver
from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import (
    fill_transform_msg, inv, rot_angle, T_to_xyzrpy, transform_msg_to_T)
from mycobot_calibration.kinematics import Chain
from mycobot_calibration.markers import board_marker, camera_pose_markers, delete_all
from mycobot_calibration.pose_planner import PosePlanner, default_params, nominal_board_pose
from mycobot_calibration.util import default_board_file, default_description_calibration


def _T_list(T):
    return [[float(v) for v in row] for row in T]


class HandEyeCalibrator(Node):

    def __init__(self):
        super().__init__('handeye_calibrator')
        p = self.declare_parameter
        p('board_file', default_board_file())
        p('base_frame', 'base')
        p('flange_frame', 'link6')
        p('camera_frame', 'camera_optical_frame')   # URDF nominal camera frame
        p('tcp_frame', 'gripper_tcp')
        p('image_topic', 'camera/image_raw')
        p('camera_info_topic', 'camera/camera_info')
        p('arm_action', 'arm_controller/follow_joint_trajectory')
        p('poses_file', '')
        p('output_dir', os.path.expanduser('~/handeye_calib'))
        p('base_calibration_file', default_description_calibration())
        p('auto_start', False)
        p('settle_time', 1.0)
        p('min_corners', 12)
        p('max_pnp_rms_px', 1.0)
        p('return_to_start', True)
        # board guess used when it is not visible at start (centre in base, lying flat)
        p('board_guess_center_xyz', [0.0, -0.22, 0.0])
        p('board_guess_yaw', 0.0)
        for k, v in default_params().items():
            if k not in ('tool_points',):
                p('planner.' + k, v)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g

        self.target = CharucoTarget(g('board_file'))
        self.base, self.flange = g('base_frame'), g('flange_frame')
        self.cam_frame, self.tcp_frame = g('camera_frame'), g('tcp_frame')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_img = None          # (stamp_sec, cv image)
        self.K = self.D = None
        self.img_size = None
        self.joint_state = {}
        self.urdf = None
        self.arm = None
        self.plan_poses = []            # list of (T_base_cam or None, q)
        self.plan_status = []
        self.samples = []
        self.T_base_board_est = None
        self.result = None
        self._abort = threading.Event()
        self._worker = None
        self._last_preview = 0.0
        self.run_dir = None

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sens = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, g('image_topic'), self._on_image, 2,
                                 callback_group=sens)
        self.create_subscription(CameraInfo, g('camera_info_topic'), self._on_info, 2,
                                 callback_group=sens)
        self.create_subscription(JointState, 'joint_states', self._on_js, 10,
                                 callback_group=sens)
        self.create_subscription(String, 'robot_description', self._on_urdf, latched,
                                 callback_group=sens)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', latched)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 2)
        self.status_pub = self.create_publisher(String, '~/status', latched)

        cb = ReentrantCallbackGroup()
        self.arm_client = ActionClient(self, FollowJointTrajectory, g('arm_action'),
                                       callback_group=cb)
        for name, fn in [('plan', self._srv_plan), ('run', self._srv_run),
                         ('capture', self._srv_capture), ('solve', self._srv_solve),
                         ('clear', self._srv_clear), ('abort', self._srv_abort)]:
            self.create_service(Trigger, '~/' + name, fn, callback_group=cb)
        if g('auto_start'):
            self._auto_timer = self.create_timer(3.0, self._auto_start, callback_group=cb)
        self.status('ready - call ~/plan, ~/run or ~/capture')

    # --- inputs ----------------------------------------------------------
    def _on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.latest_img = (stamp, img)
        now = time.monotonic()
        busy = self._worker is not None and self._worker.is_alive()
        if self.K is not None and not busy and now - self._last_preview > 0.5:
            self._last_preview = now        # live preview while idle
            det = self.target.detect(img, self.K, self.D, self.g('min_corners'))
            self._publish_debug(img, det, msg.header)

    def _on_info(self, msg):
        if self.K is None:
            self.get_logger().info(f'camera_info received ({msg.width}x{msg.height})')
        self.K = np.array(msg.k, float).reshape(3, 3)
        self.D = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
        self.img_size = (msg.width, msg.height)

    def _on_js(self, msg):
        with self.lock:
            for n, v in zip(msg.name, msg.position):
                self.joint_state[n] = v

    def _on_urdf(self, msg):
        self.urdf = msg.data
        self.arm = Chain(self.urdf, self.base, self.flange)
        self.get_logger().info(f'robot_description received, arm joints {self.arm.joint_names}')

    # --- helpers ---------------------------------------------------------
    def status(self, text, level='info'):
        # rclpy forbids different severities from the same call site -> one call per level
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
                self.tf_buf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def current_q(self):
        with self.lock:
            if self.arm is None or not all(n in self.joint_state for n in self.arm.joint_names):
                return None
            return np.array([self.joint_state[n] for n in self.arm.joint_names])

    def wait_ready(self, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            missing = []
            if self.arm is None:
                missing.append('robot_description')
            if self.K is None:
                missing.append('camera_info')
            if self.latest_img is None:
                missing.append('image')
            if self.current_q() is None:
                missing.append('joint_states')
            if self.lookup(self.flange, self.cam_frame) is None:
                missing.append(f'TF {self.flange}->{self.cam_frame}')
            if not missing:
                return True
            time.sleep(0.2)
        self.status(f'not ready, missing: {missing}', 'error')
        return False

    def fresh_image(self, after, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                li = self.latest_img
            if li is not None and li[0] >= after:
                return li[1]
            time.sleep(0.03)
        return None

    def ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_debug(self, img, det, header=None):
        vis = self.target.draw(img, det, self.K, self.D)
        msg = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
        if header is not None:
            msg.header = header
        self.debug_pub.publish(msg)

    # --- markers ---------------------------------------------------------
    def publish_markers(self):
        ma = MarkerArray()
        ma.markers.append(delete_all(self.base, 'poses'))
        if self.T_base_board_est is not None:
            ma.markers += board_marker(self.target, self.T_base_board_est, self.base,
                                       'board_estimate', alpha=0.5)
        for i, ((T_cam, _), st) in enumerate(zip(self.plan_poses, self.plan_status)):
            if T_cam is not None:
                ma.markers += camera_pose_markers(T_cam, self.base, i, st)
        if self.result is not None:
            ma.markers += board_marker(self.target, self.result['T_base_board'], self.base,
                                       'board_calibrated', tint=(0.2, 0.9, 0.3, 0.35))
        self.marker_pub.publish(ma)

    # --- planning --------------------------------------------------------
    def estimate_board(self):
        X_nom = self.lookup(self.flange, self.cam_frame)
        T_bl6 = self.lookup(self.base, self.flange)
        img = self.fresh_image(self.ros_now() - 0.5)
        if img is not None and T_bl6 is not None:
            det = self.target.detect(img, self.K, self.D, self.g('min_corners'))
            self._publish_debug(img, det)
            if det['ok']:
                self.T_base_board_est = T_bl6 @ X_nom @ det['T_cam_board']
                self.status('board found in the current image (pose via nominal X)')
                return True
        T = nominal_board_pose(np.zeros(3), float(self.g('board_guess_yaw')))
        c = T[:3, :3] @ self.target.center + T[:3, 3]
        T[:3, 3] += np.array(self.g('board_guess_center_xyz'), float) - c
        self.T_base_board_est = T
        self.status('board not visible: using board_guess_center_xyz / board_guess_yaw', 'warn')
        return False

    def make_plan(self):
        if not self.wait_ready():
            return False
        q0 = self.current_q()
        self.start_q = q0
        pf = self.g('poses_file')
        X_nom = self.lookup(self.flange, self.cam_frame)
        self.estimate_board()
        if pf:
            with open(os.path.expanduser(pf)) as f:
                qs = [np.array(q, float) for q in yaml.safe_load(f)['joint_positions']]
            self.plan_poses = [(self.arm.fk(q) @ X_nom, q) for q in qs]
            self.status(f'loaded {len(qs)} poses from {pf}')
        else:
            params = {k: self.g('planner.' + k) for k in default_params() if k != 'tool_points'}
            T_tcp = self.lookup(self.flange, self.tcp_frame)
            params['tool_points'] = [T_tcp] if T_tcp is not None else []
            params['ignore_links'] = list(params['ignore_links'])
            self.status('planning calibration poses (IK on the URDF, ~20-40 s) ...')
            planner = PosePlanner(self.arm, self.target, X_nom, self.K, self.D,
                                  self.img_size, params)
            poses, n_feasible = planner.plan(self.T_base_board_est, q0)
            self.plan_poses = poses
            if not poses:
                self.status('no feasible pose: move the board closer / check board guess, '
                            'planner.min_z, planner.dist_*', 'error')
                self.publish_markers()
                return False
            Tg = [self.arm.fk(q) for _, q in poses]
            mx, med = solver.motion_diversity(Tg)
            self.status(f'planned {len(poses)} poses ({n_feasible} feasible candidates), '
                        f'relative rotation max {mx:.0f} deg / median {med:.0f} deg')
        self.plan_status = ['planned'] * len(self.plan_poses)
        self.publish_markers()
        return True

    # --- motion ----------------------------------------------------------
    def move_to(self, q, timeout=60.0):
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            self.status('arm action server not available', 'error')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.arm.joint_names)
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(v) for v in q], time_from_start=Duration(sec=2))]
        done = threading.Event()
        holder = {}

        def on_result(f):
            holder['res'] = f.result().result
            done.set()

        def on_goal(f):
            gh = f.result()
            if not gh.accepted:
                holder['res'] = None
                done.set()
                return
            gh.get_result_async().add_done_callback(on_result)

        self.arm_client.send_goal_async(goal).add_done_callback(on_goal)
        if not done.wait(timeout):
            self.status('motion timed out', 'error')
            return False
        res = holder.get('res')
        if res is None or res.error_code != 0:
            self.status(f'motion failed: {None if res is None else res.error_string}', 'error')
            return False
        return True

    # --- capture ---------------------------------------------------------
    def capture_here(self, idx=None):
        time.sleep(float(self.g('settle_time')))
        img = self.fresh_image(self.ros_now())
        T_bl6 = self.lookup(self.base, self.flange)
        q = self.current_q()
        if img is None or T_bl6 is None:
            self.status('no fresh image / TF', 'error')
            return False
        det = self.target.detect(img, self.K, self.D, self.g('min_corners'))
        self._publish_debug(img, det)
        if not det['ok'] or det['rms'] > self.g('max_pnp_rms_px'):
            why = (f"{det['n']} corners" if not det['ok'] else f"PnP rms {det['rms']:.2f}px")
            self.status(f'sample rejected ({why})', 'warn')
            return False
        k = len(self.samples)
        self.samples.append({
            'T_base_link6': T_bl6, 'T_cam_board': det['T_cam_board'],
            'ids': det['ids'].reshape(-1), 'pixels': det['corners'].reshape(-1, 2).astype(float),
            'K': self.K.copy(), 'D': self.D.copy(), 'q': q, 'pose_index': idx,
            'pnp_rms': det['rms']})
        self._ensure_run_dir()
        cv2.imwrite(os.path.join(self.run_dir, 'images', f'{k:02d}.png'), img)
        self.status(f"sample {k} ok: {det['n']}/{self.target.n_corners} corners, "
                    f"PnP rms {det['rms']:.2f}px")
        return True

    def _ensure_run_dir(self):
        if self.run_dir is None:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = os.path.join(os.path.expanduser(self.g('output_dir')), stamp)
            os.makedirs(os.path.join(self.run_dir, 'images'), exist_ok=True)
            self.get_logger().info(f'saving to {self.run_dir}')

    # --- solve -----------------------------------------------------------
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
        X_nom = self.lookup(self.flange, self.cam_frame)
        xyzrpy = T_to_xyzrpy(X)
        lines = [f"hand-eye result ({r['n_samples']} samples, method {r['method']}):",
                 f"  link6 -> camera_optical  xyz [m] {np.round(xyzrpy[:3], 5).tolist()}",
                 f"                           rpy [rad] {np.round(xyzrpy[3:], 5).tolist()}",
                 f"  base->board scatter: {r['board_trans_rms_mm']:.2f} mm, "
                 f"{r['board_rot_rms_deg']:.3f} deg (rms)",
                 f"  chain reprojection rms: {r['chain_reproj_rms_px']:.2f} px"]
        for name, c in r['candidates'].items():
            e = inv(X) @ c['X']
            lines.append(f"    {name:10s} scatter {c['board_trans_rms_mm']:6.2f} mm  "
                         f"diff to final {np.linalg.norm(e[:3, 3]) * 1000:5.2f} mm "
                         f"{np.degrees(rot_angle(e[:3, :3])):5.2f} deg")
        if X_nom is not None:
            e = inv(X_nom) @ X
            lines.append(f"  vs URDF nominal: {np.linalg.norm(e[:3, 3]) * 1000:.2f} mm, "
                         f"{np.degrees(rot_angle(e[:3, :3])):.2f} deg")
        X_true = self.lookup(self.flange, self.cam_frame + '_true')
        if X_true is not None:
            e = inv(X_true) @ X
            lines.append(f"  vs SIM GROUND TRUTH: {np.linalg.norm(e[:3, 3]) * 1000:.2f} mm, "
                         f"{np.degrees(rot_angle(e[:3, :3])):.3f} deg")
        for ln in lines:
            self.get_logger().info(ln)
        self.status(lines[0] + f" scatter {r['board_trans_rms_mm']:.2f} mm")

        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.flange
        ts.child_frame_id = self.cam_frame + '_calib'
        fill_transform_msg(ts.transform, X)
        self.static_tf.sendTransform(ts)
        self.publish_markers()
        self._save(r, xyzrpy, lines)
        return True

    def _save(self, r, xyzrpy, lines):
        self._ensure_run_dir()
        d = self.run_dir
        with open(os.path.join(d, 'samples.yaml'), 'w') as f:
            yaml.safe_dump({'samples': [{
                'q': [float(v) for v in s['q']] if s['q'] is not None else None,
                'T_base_link6': _T_list(s['T_base_link6']),
                'T_cam_board': _T_list(s['T_cam_board']),
                'ids': [int(v) for v in s['ids']],
                'pixels': [[float(a), float(b)] for a, b in s['pixels']],
                'pnp_rms_px': float(s['pnp_rms'])} for s in self.samples],
                'K': _T_list(self.samples[0]['K']), 'D': [float(v) for v in self.samples[0]['D']],
                'board_file': self.g('board_file')}, f, sort_keys=False)
        with open(os.path.join(d, 'result.yaml'), 'w') as f:
            yaml.safe_dump({
                'method': r['method'], 'n_samples': r['n_samples'],
                'link6_to_camera_optical': {'T': _T_list(r['X'])},
                'T_base_board': _T_list(r['T_base_board']),
                'board_trans_rms_mm': r['board_trans_rms_mm'],
                'board_rot_rms_deg': r['board_rot_rms_deg'],
                'board_trans_per_sample_mm': r['board_trans_per_sample_mm'],
                'chain_reproj_rms_px': r['chain_reproj_rms_px'],
                'chain_reproj_per_sample_px': r['chain_reproj_per_sample_px'],
                'candidates': {k: {'board_trans_rms_mm': float(v['board_trans_rms_mm']),
                                   'board_rot_rms_deg': float(v['board_rot_rms_deg']),
                                   'T': _T_list(v['X'])} for k, v in r['candidates'].items()},
                'log': lines}, f, sort_keys=False)
        # drop-in calibration file for mycobot_description
        try:
            with open(self.g('base_calibration_file')) as f:
                base = yaml.safe_load(f)
        except OSError:
            base = {}
        base['link6_to_camera_optical'] = dict(zip(
            ['x', 'y', 'z', 'roll', 'pitch', 'yaw'], [round(float(v), 6) for v in xyzrpy]))
        with open(os.path.join(d, 'hand_eye.yaml'), 'w') as f:
            f.write(f'# hand-eye calibration {os.path.basename(d)}, method {r["method"]}, '
                    f'{r["n_samples"]} samples, board scatter {r["board_trans_rms_mm"]:.2f} mm\n')
            yaml.safe_dump(base, f, sort_keys=False)
        if self.plan_poses:
            with open(os.path.join(d, 'planned_poses.yaml'), 'w') as f:
                yaml.safe_dump({'joint_names': self.arm.joint_names, 'joint_positions': [
                    [float(v) for v in q] for _, q in self.plan_poses]}, f)
        self.status(f'saved: {d}/hand_eye.yaml (use calibration_file:=...)')

    # --- run -------------------------------------------------------------
    def _run(self):
        self._abort.clear()
        if not self.plan_poses and not self.make_plan():
            return
        self.samples, self.result, self.run_dir = [], None, None
        n = len(self.plan_poses)
        for i, (_, q) in enumerate(self.plan_poses):
            if self._abort.is_set():
                self.status('aborted', 'warn')
                break
            self.plan_status[i] = 'current'
            self.publish_markers()
            self.status(f'pose {i + 1}/{n}: moving')
            ok = self.move_to(q) and self.capture_here(i)
            self.plan_status[i] = 'ok' if ok else 'failed'
            self.publish_markers()
        if self.g('return_to_start') and getattr(self, 'start_q', None) is not None \
                and not self._abort.is_set():
            self.status('returning to the start pose')
            self.move_to(self.start_q)
        self.status(f'{len(self.samples)}/{n} samples captured')
        self.solve()

    def _start_worker(self, fn):
        if self._worker is not None and self._worker.is_alive():
            return False
        self._worker = threading.Thread(target=fn, daemon=True)
        self._worker.start()
        return True

    def _auto_start(self):
        self._auto_timer.cancel()
        self._start_worker(self._run)

    # --- services --------------------------------------------------------
    def _srv_plan(self, req, res):
        res.success = self._start_worker(self.make_plan)
        res.message = 'planning (see ~/status and RViz)' if res.success else 'busy'
        return res

    def _srv_run(self, req, res):
        res.success = self._start_worker(self._run)
        res.message = 'running (see ~/status and RViz)' if res.success else 'busy'
        return res

    def _srv_capture(self, req, res):
        if self._worker is not None and self._worker.is_alive():
            res.success, res.message = False, 'busy'
            return res
        if not self.wait_ready(timeout=3.0):
            res.success, res.message = False, 'not ready'
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
        self.samples, self.plan_poses, self.plan_status = [], [], []
        self.result, self.run_dir = None, None
        self.publish_markers()
        res.success, res.message = True, 'cleared'
        return res

    def _srv_abort(self, req, res):
        self._abort.set()
        res.success, res.message = True, 'abort requested (after the current pose)'
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
