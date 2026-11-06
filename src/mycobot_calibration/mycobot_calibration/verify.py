"""Checks of the camera -> robot base transform (after hand-eye calibration).

Launch with the calibrated URDF (calibration_file:=<run>/hand_eye.yaml): both checks use
TF base -> camera_optical_frame, i.e. exactly what an application would use.

  ~/verify_views  move to NEW viewpoints (other seed than the calibration), detect the board
                  in each and compute every board corner in the base frame. The board did not
                  move, so all views must agree: the spread [mm] is the real-world error of
                  "pixel -> base" for that board (independent of the calibration samples).
  ~/verify_touch  detect the board from the current pose, then move the gripper TCP down to
                  touch_hover above 4 board corners (approach from touch_approach above, slow),
                  pausing touch_pause s at each so the gap can be checked with a ruler.
                  Errors also include the TCP offset (link6_to_gripper_tcp, an estimate).
"""
import datetime
import os
import time

import numpy as np
import yaml
from std_srvs.srv import Trigger

from mycobot_calibration.geometry import (
    T_to_xyzrpy, make_T, rot_angle, rot_z, xyzrpy_to_T)


def _mean_T(Ts):
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    return make_T(U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt,
                  np.mean([T[:3, 3] for T in Ts], axis=0))


class VerifyMixin:

    def _init_verify(self, cb):
        p = self.declare_parameter
        p('verify_n_poses', 8)
        p('verify_seed_offset', 100)            # verify seed = seed + this (new viewpoints)
        # pass: corner spread over the views. The arm itself repeats to ~3-6 mm (servo error
        # ~1-3 deg), so a tighter max fails on arm error, not on the calibration.
        p('verify_max_corner_mm', 6.0)          # max corner spread
        p('verify_max_rms_mm', 3.0)             # rms corner spread
        p('touch_hover', 0.04)                  # TCP height above the board at the check [m]
        p('touch_approach', 0.08)               # approach height [m]
        p('touch_pause', 8.0)                   # time to measure at each corner [s]
        p('touch_velocity_scaling', 0.1)
        # the finger tip must stop within this of the hover point (FK of the MEASURED joints);
        # a servo stopping 3 deg short moves the tip ~15 mm, so joint tolerance is not enough
        p('touch_tolerance_mm', 5.0)
        self.create_service(Trigger, '~/verify_views', self._srv_verify_views, callback_group=cb)
        self.create_service(Trigger, '~/verify_touch', self._srv_verify_touch, callback_group=cb)

    # ------------------------------------------------------------ helpers
    def _corners_base(self, T_base_board, ids=None):
        P = self.target.corners3d if ids is None else self.target.corners3d[ids]
        return (T_base_board[:3, :3] @ P.T).T + T_base_board[:3, 3]

    def _board_here(self, n_frames=5):
        """T_base_board from the current pose: mean of n fresh detections (or None)."""
        time.sleep(float(self.g('settle_time')))
        Ts = []
        for _ in range(n_frames * 2):
            img = self.fresh_image(self.now())
            T_bc = self.lookup(self.base, self.cam)
            if img is None or T_bc is None or self.current_q() is None:   # stale joints
                continue
            det = self.target.detect(img, self.K, self.D, self.g('min_corners'))
            self._publish_debug(img, det)
            if det['ok'] and det['rms'] <= self.g('max_pnp_rms_px'):
                Ts.append(T_bc @ det['T_cam_board'])
            if len(Ts) >= n_frames:
                break
        return _mean_T(Ts) if Ts else None

    def _which_camera_pose(self):
        """Log the link6 -> camera transform in use; warn when it is the uncalibrated one."""
        X = self.lookup(self.flange, self.cam)
        v = T_to_xyzrpy(X)
        self.status(f'camera pose in the URDF (link6 -> {self.cam}): xyz [mm] '
                    f'{np.round(v[:3] * 1000, 1).tolist()} rpy [deg] '
                    f'{np.round(np.degrees(v[3:]), 2).tolist()}')
        try:
            with open(os.path.expanduser(self.g('base_calibration_file'))) as f:
                c = yaml.safe_load(f)['link6_to_camera_optical']
            nom = xyzrpy_to_T([c[k] for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')])
            if np.linalg.norm(nom[:3, 3] - X[:3, 3]) < 1e-4 and \
                    rot_angle(nom[:3, :3].T @ X[:3, :3]) < 1e-4:
                self.status('the URDF still has the NOMINAL camera pose: launch with '
                            'calibration_file:=<run>/hand_eye.yaml to check the calibration',
                            'warn')
        except Exception:  # noqa: BLE001
            pass

    def _truth_board(self):
        """Simulation only: TF base -> charuco_board_true published by sim_camera."""
        return self.lookup(self.base, 'charuco_board_true')

    # ------------------------------------------------------- verify views
    def _verify_views(self):
        self._abort.clear()
        if not self.wait_ready():
            return
        g = self.g
        self._which_camera_pose()
        keep = (self.plan, self.plan_status, self.start_q)
        n = int(g('verify_n_poses'))
        if not self.make_plan(seed=int(g('seed')) + int(g('verify_seed_offset')), n_poses=n,
                              use_poses_file=False):
            self.plan, self.plan_status, self.start_q = keep
            return
        vplan, start = self.plan, self.start_q
        boards = []
        for i, (_, q) in enumerate(vplan):
            if self._abort.is_set():
                self.status('verify aborted', 'warn')
                break
            self.plan_status[i] = 'current'
            self.publish_markers()
            ok, msg = self._move(q)
            if not ok:
                self.status(f'motion failed ({msg}), checking where the arm stopped', 'warn')
            ok = self._at_pose(q, i, len(vplan))          # measured joints are used anyway
            T = self._board_here(3) if ok else None
            self.plan_status[i] = 'ok' if T is not None else 'failed'
            self.publish_markers()
            if T is not None:
                boards.append(T)
                c = self._corners_base(T).mean(0)
                self.status(f'verify view {i + 1}/{len(vplan)}: board centre in base [mm] '
                            f'{np.round(c * 1000, 1).tolist()}')
            else:
                self.status(f'verify view {i + 1}/{len(vplan)}: no board ({msg})', 'warn')
        if start is not None and not self._abort.is_set():
            self._move(start)
        self.plan, self.plan_status, self.start_q = keep
        self.publish_markers()
        if len(boards) < 3:
            self.status(f'verify_views: only {len(boards)} views with the board', 'error')
            return

        P = np.array([self._corners_base(T) for T in boards])        # views x corners x 3
        dev = np.linalg.norm(P - P.mean(0), axis=2) * 1000            # mm
        Tm = _mean_T(boards)
        rot = [np.degrees(rot_angle(Tm[:3, :3].T @ T[:3, :3])) for T in boards]
        per_view = dev.max(1)
        lines = [f'verify_views: {len(boards)} new viewpoints, same board corner seen in base:',
                 f'  corner spread rms {np.sqrt(np.mean(dev ** 2)):.2f} mm, max {dev.max():.2f} mm'
                 f', board tilt spread max {max(rot):.2f} deg',
                 '  worst corner per view [mm]: ' + ', '.join(f'{v:.1f}' for v in per_view)]
        if self.result is not None:
            d = np.linalg.norm(P.mean(0) - self._corners_base(self.result['T_base_board']),
                               axis=1) * 1000
            lines.append(f'  vs board pose from the calibration run: mean {d.mean():.2f} mm, '
                         f'max {d.max():.2f} mm (only meaningful if the board was not moved)')
        Tt = self._truth_board()
        if Tt is not None:
            d = np.linalg.norm(P - self._corners_base(Tt), axis=2) * 1000
            lines.append(f'  vs SIM GROUND TRUTH: mean {d.mean():.2f} mm, max {d.max():.2f} mm')
        rms = float(np.sqrt(np.mean(dev ** 2)))
        max_ok = dev.max() <= float(g('verify_max_corner_mm'))
        rms_ok = rms <= float(g('verify_max_rms_mm'))
        passed = max_ok and rms_ok
        lines.append(f"  -> {'PASS' if passed else 'FAIL'} (corner spread max {dev.max():.2f} "
                     f"{'<=' if max_ok else '>'} {g('verify_max_corner_mm')} mm, rms {rms:.2f} "
                     f"{'<=' if rms_ok else '>'} {g('verify_max_rms_mm')} mm)")
        for ln in lines:
            self.get_logger().info(ln)
        self.status(lines[-1].strip(), 'info' if passed else 'warn')
        self._save_verify('views', lines, {'board_centres_base_mm': [
            [float(v) for v in self._corners_base(T).mean(0) * 1000] for T in boards],
            'corner_spread_rms_mm': rms,
            'corner_spread_max_mm': float(dev.max()), 'pass': bool(passed)})

    # ------------------------------------------------------- verify touch
    def _tcp_down_ik(self, p, seed_q):
        """IK for the TCP at point p pointing straight down; yaw chosen for least motion."""
        best = None
        for yaw_deg in range(0, 360, 45):
            R = rot_z(np.radians(yaw_deg)) @ np.diag([1.0, -1.0, -1.0])   # z down
            q = self.moveit.compute_ik(make_T(R, p), self.g('ik_link'), self._arm_joints,
                                       seed_q, timeout=float(self.g('ik_timeout')))
            if q is not None:
                cost = float(np.max(np.abs(np.asarray(q) - seed_q)))
                if best is None or cost < best[0]:
                    best = (cost, np.asarray(q), yaw_deg)
        return best

    def _move_near(self, q, what):
        """Move; a CONTROL_FAILED (servo stopped a few deg short) still counts when the
        measured joints are within pose_tolerance_deg. Returns True when usable."""
        ok, msg = self._move(q)
        if ok:
            return True
        q_now = self.current_q()
        err = (float(np.degrees(np.max(np.abs(q_now - np.asarray(q)))))
               if q_now is not None else float('inf'))
        if err <= float(self.g('pose_tolerance_deg')):
            return True
        self.status(f'{what}: motion failed ({msg}), joints {err:.1f} deg off', 'warn')
        return False

    def _verify_touch(self):
        self._abort.clear()
        if not self.wait_ready():
            return
        g = self.g
        self._which_camera_pose()
        self.add_table()                    # desk (+ table_margin) in the planning scene
        hover = max(0.01, float(g('touch_hover')))
        min_hover = float(g('table_margin')) + 0.005 if g('add_table') else 0.0
        if hover < min_hover:
            self.status(f'touch_hover {hover * 1000:.0f} mm is inside table_margin: using '
                        f'{min_hover * 1000:.0f} mm', 'warn')
            hover = min_hover
        tol = float(g('touch_tolerance_mm')) / 1000.0
        approach = max(hover + 0.03, float(g('touch_approach')))
        q_start = self.current_q()
        T_board = self._board_here(5)
        if T_board is None:
            # move to one viewpoint above the board (board_guess_* / last estimate) and retry
            self.status('verify_touch: board not visible here, moving to a viewpoint above it')
            keep = (self.plan, self.plan_status, self.start_q)
            ok = self.make_plan(seed=int(g('seed')) + 7, n_poses=1, use_poses_file=False)
            view = self.plan[0][1] if ok and self.plan else None
            self.plan, self.plan_status, self.start_q = keep
            if view is not None and self._move(view)[0]:
                T_board = self._board_here(5)
        if T_board is None:
            self.status('verify_touch: board not visible - put the arm where the camera '
                        'sees the board and call verify_touch again', 'error')
            return
        cols, rows = self.target.cols - 1, self.target.rows - 1     # inner corner grid
        ids = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
        pts = self._corners_base(T_board, ids)
        self.status('verify_touch: board corners in base [mm]: ' + '; '.join(
            f'id {i}: {np.round(p * 1000, 1).tolist()}' for i, p in zip(ids, pts)))
        Tt = self._truth_board()
        old_v = g('velocity_scaling')
        results = []
        shown = 0
        try:
            self.set_parameters([rclpy_param('velocity_scaling', float(g('touch_velocity_scaling')))])
            for k, (cid, p) in enumerate(zip(ids, pts)):
                if self._abort.is_set():
                    break
                up = self._tcp_down_ik(p + [0, 0, approach], self.current_q())
                if up is None:
                    self.status(f'corner id {cid}: not reachable from above, skipped', 'warn')
                    continue
                down = self._tcp_down_ik(p + [0, 0, hover], up[1])
                if down is None:
                    self.status(f'corner id {cid}: hover pose not reachable, skipped', 'warn')
                    continue
                if not self._move_near(up[1], f'corner id {cid} approach'):
                    continue
                if self._abort.is_set():
                    break
                if not self._move_near(down[1], f'corner id {cid} descent'):
                    self._move(up[1])
                    continue
                time.sleep(float(g('settle_time')))
                if self.current_q() is None:
                    self.status('joint_states stale: verify_touch stopped', 'error')
                    break
                T_tcp = self.lookup(self.base, g('ik_link'))
                if T_tcp is None:
                    self.status(f"corner id {cid}: no TF {self.base}->{g('ik_link')}, skipped",
                                'warn')
                    self._move(up[1])
                    continue
                tcp = T_tcp[:3, 3]
                goal = p + [0, 0, hover]
                d = tcp - goal                            # model (measured joints) vs target
                if np.linalg.norm(d) > tol:
                    self.status(f'corner id {cid}: finger tip stopped {np.linalg.norm(d) * 1000:.1f}'
                                f' mm from the hover point (z {d[2] * 1000:+.1f} mm, servo short of'
                                f' target) - skipped, > touch_tolerance_mm', 'warn')
                    self._move(up[1])
                    continue
                msg = (f'corner {k + 1}/4 (id {cid}): finger tip should be '
                       f'{(tcp[2] - p[2]) * 1000:.0f} mm right above the corner (model, measured '
                       f'joints; xy {np.linalg.norm(d[:2]) * 1000:.1f} mm) - measure the offset '
                       f'now ({g("touch_pause"):.0f} s)')
                if Tt is not None:
                    pt = self._corners_base(Tt, [cid])[0]
                    e = tcp - pt - [0, 0, hover]
                    msg += (f' | SIM truth: xy {np.linalg.norm(e[:2]) * 1000:.2f} mm, '
                            f'z {e[2] * 1000:+.2f} mm')
                    results.append(float(np.linalg.norm(e[:2]) * 1000))
                self.status(msg)
                shown += 1
                t0 = time.monotonic()
                while time.monotonic() - t0 < float(g('touch_pause')) and not self._abort.is_set():
                    time.sleep(0.1)
                self._move(up[1])
        finally:
            self.set_parameters([rclpy_param('velocity_scaling', float(old_v))])
        if q_start is not None and not self._abort.is_set():
            self._move(q_start)
        done = f'verify_touch done: {shown}/4 corners shown' + (
            f' (SIM xy error max {max(results):.2f} mm)' if results else '')
        self.status(done)

    def _save_verify(self, kind, lines, data):
        d = os.path.expanduser(self.g('output_dir'))
        os.makedirs(d, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(d, f'verify_{kind}_{stamp}.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(dict(data, log=lines), f, sort_keys=False)
        self.get_logger().info(f'saved {path}')

    # ----------------------------------------------------------- services
    def _srv_verify_views(self, req, res):
        res.success = self._start(self._verify_views)
        res.message = 'verifying (see ~/status)' if res.success else 'busy'
        return res

    def _srv_verify_touch(self, req, res):
        res.success = self._start(self._verify_touch)
        res.message = 'touch check running (see ~/status)' if res.success else 'busy'
        return res


def rclpy_param(name, value):
    from rclpy.parameter import Parameter
    return Parameter(name, value=value)
