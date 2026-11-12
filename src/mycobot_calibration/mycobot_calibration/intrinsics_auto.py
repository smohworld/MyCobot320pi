"""Automatic camera-intrinsics capture: the arm moves the flange camera over the ChArUco board
lying on the desk until the image is covered, then runs the intrinsics calibration.

Why: the board has to reach every part of the image (corners/edges too) and be seen tilted.
By hand the board stays near the centre (coverage ~60%). Here the camera poses are chosen to
fill the empty cells of an 8x6 image grid (the "coverage" of intrinsics_calib), closed loop:

  round: board pose (current image / board_guess_xyz) -> pool of camera poses (tilt, azimuth,
         distance, roll, aim point pushed past the board edge so the board lands at the image
         border) -> predict which grid cells the board corners would fall in (pinhole, rough
         K) -> greedy pick (most new cells, tilted views until 8, then variety) with MoveIt IK
         -> move + settle + capture (corner count, sharpness) -> real coverage
  repeat up to max_rounds until coverage >= target_coverage, enough tilted views and
  >= min_views, then intrinsics_calib.report_and_save() writes <output_dir>/<time>/camera.yaml.

  ros2 launch mycobot_calibration intrinsics_auto.launch.py connection:=serial \
      serial_port:=/dev/ttyUSB0 board_xyz:=0.22,0.0,-0.007
  ros2 service call /intrinsics_auto/run std_srvs/srv/Trigger     # moves the arm!
  ros2 service call /intrinsics_auto/abort std_srvs/srv/Trigger
  ros2 service call /intrinsics_auto/calibrate std_srvs/srv/Trigger   # with the views so far

Start with the camera looking at the board (goto_lookdown): IK solutions must stay within
max_joint_dev_deg of the start pose, like the hand-eye node. Install the result with
  cp <output_dir>/<time>/camera.yaml <ws>/src/mycobot_calibration/config/camera.yaml
(and rebuild) - this node never overwrites config/camera.yaml itself.
"""
import datetime
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import inv, look_at, rot_z, transform_msg_to_T
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.intrinsics import load_intrinsics
from mycobot_calibration.intrinsics_calib import (
    F_GUESS, MIN_TILT_DEG, MIN_TILTED_VIEWS, calibrate, coverage, detect, report_and_save, tilt_deg)
from mycobot_calibration.intrinsics_planner import GRID, ViewPlanner, cells_of
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_calibration.pose_generator import (
    ViewpointGenerator, board_pose_flat, order_by_joint_distance)


class IntrinsicsAuto(Node):

    def __init__(self):
        super().__init__('intrinsics_auto')
        share = get_package_share_directory('mycobot_calibration')
        p = self.declare_parameter
        p('board_file', os.path.join(share, 'config', 'charuco_board.yaml'))
        p('prior_intrinsics_file', '')        # only to PREDICT visibility; '' = pinhole f=focal_px_guess
        p('focal_px_guess', float(F_GUESS))
        p('image_topic', '/camera/image_raw')
        p('base_frame', 'base')
        p('flange_frame', 'link6')
        p('camera_frame', 'camera_optical_frame')
        p('ik_link', 'gripper_tcp')
        p('group', 'arm')
        p('board_guess_xyz', [0.22, 0.0, -0.007])   # board centre in base [m] (desk = -7 mm)
        p('board_guess_yaw', 0.0)
        # goal
        p('target_coverage', 0.85)            # fraction of the 8x6 image grid
        p('min_views', 25)
        p('max_views', 45)
        p('views_per_round', 14)
        p('max_rounds', 5)
        p('plan_margin', 0.12)                # plan until PREDICTED coverage >= target + this (the
                                              # prediction is optimistic: real captures fail more)
        # viewpoint pool
        p('pool_size', 2000)
        p('distance_min', 0.22)               # camera -> board centre [m]
        p('distance_max', 0.34)
        p('tilt_min_deg', 8.0)
        p('tilt_max_deg', 40.0)
        p('roll_range_deg', 25.0)
        # aim point jitter in the board plane, in half-board-sizes (orientation variety)
        p('aim_range', 0.25)
        p('shift_range', 0.6)                 # camera sideways shift, fraction of the distance
        p('max_incidence_deg', 55.0)
        p('image_margin_px', 8)
        p('min_visible_fraction', 0.4)        # of the chessboard corners, predicted detectable
        p('min_corners', 10)                  # captured view needs this many corners
        p('min_sharpness', 60.0)              # Laplacian variance
        p('min_camera_height', 0.08)          # above table_z [m]
        # camera position must lie in a shell around the shoulder axis: farther than reach_max the
        # arm cannot reach (0% IK success), closer than reach_min the pose needs a big rotation
        # from the start (joint_window). Measured on this arm (600 IK trials, sim 2026-11-11).
        p('shoulder_xyz', [0.0, 0.0, 0.181])
        p('reach_min', 0.20)
        p('reach_max', 0.30)                   # 1200 IK trials: 7% ok in 0.30-0.34, 49% in 0.20-0.25
        p('max_camera_height', 0.35)          # above table_z [m]; 6% ok above 0.35
        p('max_ik', 200)                      # IK calls per round
        p('max_reachable', 80)                # stop testing after this many reachable poses
        p('ik_timeout', 0.3)
        p('max_joint_dev_deg', [60.0, 90.0, 70.0, 90.0, 90.0, 90.0])
        p('seed', 0)
        p('sweep_xs', [0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30])   # ~/sweep: board x [m]
        p('sweep_ys', [0.0, 0.05])
        p('sweep_rounds', 2)
        p('lookdown_heights', [0.28, 0.24, 0.32, 0.20])   # camera height above the board [m], in order
        p('sweep_max_ik', 120)
        # MoveIt
        p('add_table', True)
        p('table_z', -0.007)
        p('table_margin', 0.012)
        p('table_size_xy', [1.5, 1.5])
        p('table_center_xy', [0.0, 0.0])
        p('velocity_scaling', 0.3)
        p('accel_scaling', 0.3)
        p('planning_time', 5.0)
        p('settle_time', 1.0)
        p('pose_tolerance_deg', 6.0)
        p('joint_state_timeout', 1.5)
        p('return_to_start', True)
        p('auto_start', False)
        p('output_dir', os.path.expanduser('~/camera_calib_auto'))
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        g = self.g

        self.target = CharucoTarget(os.path.expanduser(g('board_file')))
        self.get_logger().info(f"board: {self.target.describe()}")
        self.base, self.flange, self.cam = g('base_frame'), g('flange_frame'), g('camera_frame')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = None                    # (stamp, bgr image)
        self.js, self.js_t = {}, None
        self.views, self.names, self.tilts = [], [], []
        self.size = None
        self.run_dir = None
        self.T_board = self.start_q = self.X = self.T_l6_ik = None
        self._abort = threading.Event()
        self._worker = None
        self._last_preview = 0.0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        sens = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, g('image_topic'), self._on_image, 2, callback_group=sens)
        self.create_subscription(JointState, 'joint_states', self._on_js, 10, callback_group=sens)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 2)
        self.status_pub = self.create_publisher(String, '~/status', latched)
        self.moveit = MoveItClient(self, g('group'), self.base)
        self.planner = ViewPlanner(self.target, g, self.moveit, self.status_log)
        if g('prior_intrinsics_file'):
            self.planner.K_prior, self.planner.D_prior, _ = load_intrinsics(
                os.path.expanduser(g('prior_intrinsics_file')))
        cb = ReentrantCallbackGroup()
        for name, fn in [('run', self._srv_run), ('abort', self._srv_abort),
                         ('calibrate', self._srv_calibrate), ('clear', self._srv_clear),
                         ('sweep', self._srv_sweep)]:
            self.create_service(Trigger, '~/' + name, fn, callback_group=cb)
        if g('auto_start'):
            self._auto = self.create_timer(3.0, self._auto_start, callback_group=cb)
        self.status('ready: call ~/run (moves the arm), ~/calibrate or ~/clear')

    # ------------------------------------------------------------- inputs
    def _on_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.latest = (stamp, img)
        now = time.monotonic()
        if now - self._last_preview > 0.5:                 # live view in RViz before / during a run
            self._last_preview = now
            d = detect(self.target, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            n = 0 if d is None else len(d[0])
            self._debug(img, f'live: {n}/{self.target.n_corners} corners  views {len(self.views)}'
                             f'  cov {self.cov() * 100:.0f}%', d)

    def _on_js(self, msg):
        with self.lock:
            self.js.update(zip(msg.name, msg.position))
            if all(n in msg.name for n in ARM_JOINTS):
                self.js_t = time.monotonic()

    def status(self, text, level='info'):
        log = self.get_logger()
        if level == 'error':
            log.error(text)
        elif level == 'warn':
            log.warn(text)
        else:
            log.info(text)
        self.status_pub.publish(String(data=text))

    def status_log(self, text, level='info'):
        self.status(text, level)

    def lookup(self, parent, child):
        try:
            return transform_msg_to_T(
                self.tf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def current_q(self):
        with self.lock:
            if self.js_t is None or time.monotonic() - self.js_t > self.g('joint_state_timeout'):
                return None
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

    def wait_ready(self, timeout=15.0):
        t0 = time.monotonic()
        missing = []
        while time.monotonic() - t0 < timeout:
            missing = []
            if self.latest is None:
                missing.append(f"image on {self.g('image_topic')}")
            if self.current_q() is None:
                missing.append('joint_states')
            if self.lookup(self.flange, self.cam) is None:
                missing.append(f'TF {self.flange}->{self.cam}')
            if self.lookup(self.flange, self.g('ik_link')) is None:
                missing.append(f"TF {self.flange}->{self.g('ik_link')}")
            if not self.moveit.wait_ready(timeout=0.5):
                missing.append('move_group (/compute_ik, /move_action)')
            if not missing:
                if self.size is None:
                    self.size = self.latest[1].shape[1::-1]
                return True
            time.sleep(0.3)
        self.status(f'not ready, missing: {missing}', 'error')
        return False

    # ---------------------------------------------------------- geometry
    def add_table(self):
        g = self.g
        if not g('add_table'):
            return
        z = float(g('table_z')) + float(g('table_margin'))
        cx, cy = (float(v) for v in g('table_center_xy'))
        sx, sy = (float(v) for v in g('table_size_xy'))
        ok = self.moveit.add_box('calibration_table', [cx, cy, z - 0.025], [sx, sy, 0.05])
        if not ok:
            self.status('could not add the table to the planning scene', 'warn')

    def estimate_board(self):
        """Board pose in base: from the current image (nominal hand-eye) else board_guess_*."""
        T_bl6 = self.lookup(self.base, self.flange)
        img = self.fresh_image(self.now() - 0.5)
        if img is not None and T_bl6 is not None:
            K, D = self.planner.K_pred()
            det = self.target.detect(img, K, D, 8)
            if det['ok']:
                self.T_board = T_bl6 @ self.X @ det['T_cam_board']
                c = self.T_board[:3, :3] @ self.target.center + self.T_board[:3, 3]
                self.status(f'board seen at {np.round(c, 3).tolist()} (base frame)')
                return
        self.T_board = board_pose_flat(self.target, self.g('board_guess_xyz'),
                                       self.g('board_guess_yaw'))
        self.status('board not visible: using board_guess_xyz / board_guess_yaw', 'warn')

    # ----------------------------------------------------------- state
    def hit_cells(self):
        s = set()
        for _, img in self.views:
            s |= cells_of(img, self.size)
        return s

    def n_tilted(self):
        return sum(1 for t in self.tilts if t >= MIN_TILT_DEG)

    def cov(self):
        return coverage(self.views, self.size) if self.views else 0.0

    def done(self):
        return (self.cov() >= self.g('target_coverage') and self.n_tilted() >= MIN_TILTED_VIEWS
                and len(self.views) >= self.g('min_views'))

    # ----------------------------------------------------------- planning
    def make_plan(self):
        """One round of poses from the planner, ordered by joint distance."""
        g = self.g
        room = min(int(g('views_per_round')), int(g('max_views')) - len(self.views))
        chosen, covered, st = self.planner.plan(self.hit_cells(), self.n_tilted(),
                                                len(self.views), room, abort=self._abort)
        self.status(f"IK ok {st['ok']} / collision {st['collision']} / kinematic {st['kinematic']} / "
                    f"joint_window {st['joint_window']} ({st['reused']} poses re-used)")
        if st['ik_calls'] and st['ok'] < 0.2 * st['ik_calls']:
            self.status('few reachable poses: start with the camera over the board '
                        '(goto_lookdown), widen max_joint_dev_deg, or move the board '
                        '(board_xyz) closer to the robot', 'warn')
        q_now = self.current_q()
        return order_by_joint_distance(self.start_q if q_now is None else q_now, chosen)

    def refine_prior(self):
        """Predict with the focal length of the views so far (better aim in the next round)."""
        if len(self.views) < 8 or self.n_tilted() < 4:
            return
        try:
            _, K, _, _, _ = calibrate(self.views, self.size)
        except cv2.error:
            return
        if 0.5 * self.size[0] < K[0, 0] < 1.6 * self.size[0]:
            self.planner.K_prior, self.planner.D_prior = K, np.zeros(5)
            self.get_logger().info(f'prediction K updated: f {K[0, 0]:.0f} px')

    # ------------------------------------------------------------ capture
    def _debug(self, img, text, det=None):
        vis = img.copy()
        if det is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, det[2], det[3], (0, 255, 0))
        h, w = vis.shape[:2]
        hit = self.hit_cells()
        for r in range(GRID[1]):
            for c in range(GRID[0]):
                x0, y0 = int(c * w / GRID[0]), int(r * h / GRID[1])
                x1, y1 = int((c + 1) * w / GRID[0]), int((r + 1) * h / GRID[1])
                if (r, c) not in hit:
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

    def capture_here(self):
        g = self.g
        time.sleep(float(g('settle_time')))
        img = self.fresh_image(self.now())
        if img is None or self.current_q() is None:
            self.status('no fresh image / joint_states: view skipped', 'warn')
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        d = detect(self.target, gray)
        n = 0 if d is None else len(d[0])
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        if n < g('min_corners') or sharp < g('min_sharpness'):
            self.status(f'view rejected ({n} corners, sharpness {sharp:.0f})', 'warn')
            return False
        k = len(self.views)
        cv2.imwrite(os.path.join(self.run_dir, 'images', f'{k:03d}.png'), img)
        self.views.append((d[0], d[1]))
        self.names.append(f'{k:03d}.png')
        self.tilts.append(tilt_deg(d[0], d[1], self.size))
        self.status(f'view {k + 1}: {n} corners, tilt {self.tilts[-1]:.0f} deg, coverage '
                    f'{self.cov() * 100:.0f}%, tilted {self.n_tilted()}/{MIN_TILTED_VIEWS}')
        self._debug(img, f'views {k + 1}  cov {self.cov() * 100:.0f}%  '
                         f'tilted {self.n_tilted()}/{MIN_TILTED_VIEWS}')
        return True

    def _at_pose(self, q, timeout=3.0):
        tol = float(self.g('pose_tolerance_deg'))
        t0 = time.monotonic()
        err = float('inf')
        while time.monotonic() - t0 < timeout:
            qn = self.current_q()
            if qn is not None:
                err = float(np.degrees(np.max(np.abs(qn - np.asarray(q)))))
                if err <= tol:
                    return True
            time.sleep(0.1)
        self.status(f'joint_states {err:.1f} deg from the planned pose (> {tol:.1f}): skipped',
                    'warn')
        return False

    def _move(self, q):
        g = self.g
        return self.moveit.move_joints(ARM_JOINTS, q, g('velocity_scaling'),
                                       g('accel_scaling'), g('planning_time'))

    # ---------------------------------------------------------------- run
    def _run(self):
        self._abort.clear()
        if not self.wait_ready():
            return
        g = self.g
        self.add_table()
        self.start_q = self.current_q()
        self.X = self.lookup(self.flange, self.cam)
        self.T_l6_ik = self.lookup(self.flange, g('ik_link'))
        self.views, self.names, self.tilts = [], [], []
        self.run_dir = os.path.join(os.path.expanduser(g('output_dir')),
                                    datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(os.path.join(self.run_dir, 'images'), exist_ok=True)
        self.planner.size = self.size
        self.estimate_board()
        T_cam_now = self.lookup(self.base, self.flange) @ self.X
        self.planner.configure(self.T_board, self.X, self.T_l6_ik, self.start_q, self.size,
                               T_cam_now)
        self.planner.ik_log_path = os.path.join(self.run_dir, 'ik_log.csv')
        for rnd in range(int(g('max_rounds'))):
            if self._abort.is_set() or self.done() or len(self.views) >= g('max_views'):
                break
            self.status(f'--- round {rnd + 1}/{g("max_rounds")} ---')
            plan = self.make_plan()
            if not plan:
                self.status('no reachable pose adds coverage: stopping', 'warn')
                break
            for i, (_, q, tilt) in enumerate(plan):
                if self._abort.is_set():
                    self.status('aborted', 'warn')
                    break
                self.status(f'round {rnd + 1}: pose {i + 1}/{len(plan)} (tilt {tilt:.0f} deg)')
                ok, msg = self._move(q)
                if not ok:
                    self.status(f'motion failed ({msg}), checking where the arm stopped', 'warn')
                if self._abort.is_set():
                    break
                if self._at_pose(q) and not self.capture_here():
                    self.capture_here()            # one retry
            self.refine_prior()
            self.status(f'round {rnd + 1} done: coverage {self.cov() * 100:.0f}%, '
                        f'{len(self.views)} views, tilted {self.n_tilted()}/{MIN_TILTED_VIEWS}')
        if g('return_to_start') and not self._abort.is_set():
            ok, msg = self._move(self.start_q)
            if not ok:
                self.status(f'return to start failed ({msg})', 'warn')
        self.calibrate()

    def calibrate(self):
        if not self.views or self.size is None:
            self.status('no views', 'error')
            return False
        if self.run_dir is None:
            self.run_dir = os.path.join(os.path.expanduser(self.g('output_dir')),
                                        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self.run_dir, exist_ok=True)
        self.status(f'calibrating {len(self.views)} views, coverage {self.cov() * 100:.0f}% '
                    '(details in the terminal)')
        ok = report_and_save(self.run_dir, self.views, self.size, self.names)
        if ok:
            self.status(f'saved {self.run_dir}/camera.yaml - install: cp '
                        f'{self.run_dir}/camera.yaml <ws>/src/mycobot_calibration/config/'
                        'camera.yaml, then rebuild')
        else:
            self.status('not saved (see the terminal: rms / tilt / view count)', 'warn')
        return ok

    # -------------------------------------------------------------- sweep
    def _lookdown_q(self, c):
        """Start pose for a board centre c: camera straight above it looking down. Heights are
        tried in order (like goto_lookdown: preferred first, then others) until IK solves."""
        gen = ViewpointGenerator(self.target)
        q_cur = self.current_q()
        for h in self.g('lookdown_heights'):
            T0 = look_at(c + np.array([0, 0, h]), c, up_hint=gen.up_hint(c))
            # roll about the optical axis too (goto_lookdown lists solutions at roll 0 and 90)
            for roll in (0.0, 90.0, -90.0, 45.0, -45.0):
                T = T0.copy()
                T[:3, :3] = T0[:3, :3] @ rot_z(np.radians(roll))
                T_ik = T @ inv(self.X) @ self.T_l6_ik
                for j2, j3 in [(0, 0), (-0.6, 1.2), (0.6, -1.2), (0.6, 1.2), (-0.6, -1.2)]:
                    seed = q_cur.copy()
                    seed[1], seed[2] = j2, j3
                    q = self.moveit.compute_ik(T_ik, self.g('ik_link'), ARM_JOINTS, seed,
                                               timeout=0.3)
                    if q is not None:
                        return T, np.array(q)
        return T0, None

    def _sweep(self):
        """Plan-only evaluation of board positions (the arm does not move): for each position the
        planner runs for sweep_rounds rounds, assuming every planned view is captured."""
        g = self.g
        self._abort.clear()
        if not self.wait_ready():
            return
        self.add_table()
        self.X = self.lookup(self.flange, self.cam)
        self.T_l6_ik = self.lookup(self.flange, g('ik_link'))
        self.planner.size = self.size
        self.planner.ik_log_path = None
        out = os.path.join(os.path.expanduser(g('output_dir')), 'sweep.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rows = []
        for y in g('sweep_ys'):
            for x in g('sweep_xs'):
                if self._abort.is_set():
                    break
                c = np.array([x, y, g('table_z')])
                T_board = board_pose_flat(self.target, c, g('board_guess_yaw'))
                T_cam, q0 = self._lookdown_q(c)
                if q0 is None:
                    self.status(f'sweep x={x:.2f} y={y:.2f}: no look-down pose', 'warn')
                    rows.append((x, y, 0, 0, 0, 0.0, 0))
                    continue
                self.planner.configure(T_board, self.X, self.T_l6_ik, q0, self.size, T_cam)
                hit, tilted, n_pose, n_ik, n_ok = set(), 0, 0, 0, 0
                for _ in range(int(g('sweep_rounds'))):
                    plan, hit, st = self.planner.plan(hit, tilted, n_pose,
                                                      int(g('views_per_round')),
                                                      max_ik=int(g('sweep_max_ik')),
                                                      abort=self._abort)
                    n_ik += st['ik_calls']
                    n_ok += st['ok']
                    n_pose += len(plan)
                    tilted += sum(1 for _, _, t in plan if t >= MIN_TILT_DEG)
                cov = len(hit) / (GRID[0] * GRID[1])
                rows.append((x, y, n_pose, n_ik, n_ok, cov, tilted))
                self.status(f'SWEEP x={x:.2f} y={y:.2f}: IK ok {n_ok}/{n_ik} '
                            f'({100 * n_ok / max(n_ik, 1):.0f}%), {n_pose} poses, predicted '
                            f'coverage {cov * 100:.0f}%, tilted {tilted}')
        with open(out, 'w') as f:
            f.write('x,y,poses,ik_calls,ik_ok,pred_coverage,tilted\n')
            for r in rows:
                f.write(','.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r) + '\n')
        best = sorted(rows, key=lambda r: (-r[5], -r[4]))[:5]
        self.status('SWEEP best (x, y, poses, coverage): ' + '; '.join(
            f'({r[0]:.2f}, {r[1]:.2f}, {r[2]}, {r[5] * 100:.0f}%)' for r in best))

    def _srv_sweep(self, req, res):
        res.success = self._start(self._sweep)
        res.message = 'sweeping board positions (plan only, no motion; see ~/status)' \
            if res.success else 'busy'
        return res

    # ----------------------------------------------------------- services
    def _start(self, fn):
        if self._worker is not None and self._worker.is_alive():
            return False
        self._worker = threading.Thread(target=fn, daemon=True)
        self._worker.start()
        return True

    def _auto_start(self):
        self._auto.cancel()
        self._start(self._run)

    def _srv_run(self, req, res):
        res.success = self._start(self._run)
        res.message = 'running (moves the arm; see ~/status, ~/debug_image)' if res.success \
            else 'busy'
        return res

    def _srv_calibrate(self, req, res):
        res.success = self._start(self.calibrate)
        res.message = 'calibrating' if res.success else 'busy'
        return res

    def _srv_abort(self, req, res):
        self._abort.set()
        self.moveit.cancel()
        res.success, res.message = True, 'abort requested'
        return res

    def _srv_clear(self, req, res):
        self.views, self.names, self.tilts = [], [], []
        res.success, res.message = True, 'cleared'
        return res


def main(args=None):
    rclpy.init(args=args)
    node = IntrinsicsAuto()
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
