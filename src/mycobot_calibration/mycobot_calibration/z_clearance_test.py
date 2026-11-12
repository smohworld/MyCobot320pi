"""Desk clearance test: how far off is the finger tip in z when it descends to the desk?

For several desk points the arm descends step by step (--gaps, finger tip height above the
desk model --table-z). At each stop YOU measure the real gap between finger tips and desk
(ruler / feeler gauge / stack of paper) and type it in [mm]. The script records, per stop:
  - err_mm        = measured - commanded gap   (negative = tip LOWER than commanded: collision risk)
  - model_dz_mm   = TCP z from the URDF with the measured joints minus the goal z
                    (blind to kinematic error: only joint tracking of the driver)
  - joint error   = IK target vs measured joints [deg] (what the driver loop controls)
If err_mm is large while the joint error is small (< 0.5 deg), the error is NOT the driver
(PID would not help): it is the model / hand-eye / --table-z.

  ros2 run mycobot_calibration z_clearance_test                       # plan only (dry run)
  ros2 run mycobot_calibration z_clearance_test --go                  # fake robot
  ros2 run mycobot_calibration z_clearance_test --go --real           # REAL robot
  ros2 run mycobot_calibration z_clearance_test --go --real --bidirectional

At every prompt: number = measured gap [mm] | Enter = not measured, continue | x = skip the
rest of this point (lift) | q = lift and quit. A gap below --min-clearance ends the descent at
that point automatically. Needs move_group running (handeye.launch.py) and the driver.
Results: CSV (--out) + summary with the recommended --table-z and an x/y trend.
"""
import argparse
import csv
import datetime
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import make_T, rot_z, transform_msg_to_T
from mycobot_calibration.goto_lookdown import driver_connection
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient

DOWN = np.diag([1.0, -1.0, -1.0])
TCP = 'gripper_tcp'
DEFAULT_POINTS = '0.20,0.0;0.25,0.0;0.25,0.08;0.25,-0.08;0.30,0.0'


class Quit(Exception):
    pass


class SkipPoint(Exception):
    pass


class Probe(Node):
    def __init__(self):
        super().__init__('z_clearance_test')
        self.js = {}
        self.create_subscription(JointState, 'joint_states',
                                 lambda m: self.js.update(zip(m.name, m.position)), 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)

    def q(self):
        return np.array([self.js[n] for n in ARM_JOINTS]) if all(
            n in self.js for n in ARM_JOINTS) else None

    def T(self, parent, child):
        try:
            return transform_msg_to_T(
                self.tf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None


def parse_points(s):
    return [tuple(float(v) for v in p.split(',')) for p in s.split(';') if p.strip()]


class Test:
    def __init__(self, node, args):
        self.node, self.a = node, args
        self.mv = MoveItClient(node)
        self.conn = driver_connection(node)
        if args.go and self.conn != 'fake' and not args.real:
            raise SystemExit(f'driver connection is {self.conn!r}, not fake: add --real to move it')
        t0 = time.monotonic()
        while node.q() is None or node.T('base', TCP) is None:
            if time.monotonic() - t0 > 5.0:
                raise SystemExit(f'no joint_states / TF base -> {TCP}')
            time.sleep(0.1)
        if not self.mv.wait_ready(15.0):
            raise SystemExit('move_group not available (start handeye.launch.py first)')
        # collision box at the model desk (-2 mm): MoveIt must not reject the low stops
        self.mv.add_box('calibration_table', [0.0, 0.0, args.table_z - 0.002 - 0.025],
                        [1.5, 1.5, 0.05])
        T = node.T('base', TCP)
        R = T[:3, :3] @ DOWN
        self.yaw = args.yaw if args.yaw is not None else float(np.arctan2(R[1, 0], R[0, 0]))
        self.rows = []

    # ---------------------------------------------------------------- motion
    def ik(self, p):
        seed = self.node.q()
        best = None
        for k in range(4):
            q = self.mv.compute_ik(make_T(rot_z(self.yaw + k * np.pi / 2) @ DOWN, p), TCP,
                                   ARM_JOINTS, seed, timeout=0.1)
            if q is not None:
                cost = float(np.max(np.abs(np.asarray(q) - seed)))
                if best is None or cost < best[0]:
                    best = (cost, np.asarray(q))
        return None if best is None else best[1]

    def go_to(self, name, p, slow):
        q = self.ik(np.asarray(p, float))
        if q is None:
            raise SkipPoint(f'no IK for {name} at ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, '
                            f'{p[2] * 1000:.0f}) mm')
        if not self.a.go:
            print(f'  (dry run) {name}: z {p[2] * 1000:.1f} mm  J[deg] '
                  + ' '.join(f'{v:6.1f}' for v in np.degrees(q)))
            return q
        v = self.a.slow_velocity if slow else self.a.velocity
        ok, msg = self.mv.move_joints(ARM_JOINTS, q, v, self.a.accel)
        qn = self.node.q()
        err = np.degrees(np.max(np.abs(qn - q)))
        if not ok and err > 3.0:
            raise SkipPoint(f'{name}: motion failed ({msg}), joints {err:.1f} deg off')
        time.sleep(self.a.settle)
        return q

    # ---------------------------------------------------------------- measuring
    def ask(self, label):
        if not self.a.go:
            return None
        while True:
            s = input(f'  [{label}] measured gap tip-desk [mm] '
                      f'(Enter=skip, x=lift this point, q=quit): ').strip().lower()
            if s == '':
                return None
            if s == 'x':
                raise SkipPoint('operator skipped the rest of this point')
            if s == 'q':
                raise Quit()
            try:
                return float(s)
            except ValueError:
                print('  not a number')

    def stop_at(self, pt, idx, gap, direction):
        a = self.a
        p = np.array([pt[0], pt[1], a.table_z + gap])
        q_goal = self.go_to(f'gap {gap * 1000:.0f} mm ({direction})', p, slow=True)
        if not a.go:
            return None
        T = self.node.T('base', TCP)
        q_now = self.node.q()
        dq = np.degrees(q_now - q_goal)
        model_dz = (T[2, 3] - p[2]) * 1000 if T is not None else float('nan')
        model_dxy = (np.linalg.norm(T[:2, 3] - p[:2]) * 1000) if T is not None else float('nan')
        print(f'  commanded gap {gap * 1000:.0f} mm | model tip dz {model_dz:+.1f} mm, '
              f'dxy {model_dxy:.1f} mm | joint err max {np.max(np.abs(dq)):.2f} deg '
              + '[' + ' '.join(f'{v:+.2f}' for v in dq) + ']')
        meas = self.ask(f'pt {idx} {direction} {gap * 1000:.0f} mm')
        err = None if meas is None else meas - gap * 1000
        self.rows.append(dict(
            point=idx, x=pt[0], y=pt[1], direction=direction, gap_cmd_mm=gap * 1000,
            gap_meas_mm=meas, err_mm=err, model_dz_mm=model_dz, model_dxy_mm=model_dxy,
            joint_err_max_deg=float(np.max(np.abs(dq))),
            **{f'dq{i + 1}_deg': float(dq[i]) for i in range(6)}))
        if meas is not None and meas < a.min_clearance:
            print(f'  clearance {meas:.1f} mm < --min-clearance {a.min_clearance:.1f}: '
                  'no further descent at this point')
            raise SkipPoint('min clearance reached')
        return meas

    def run_point(self, idx, pt):
        a = self.a
        print(f'\n=== point {idx}: x {pt[0] * 1000:.0f}  y {pt[1] * 1000:.0f} mm ===')
        top = np.array([pt[0], pt[1], a.table_z + a.approach])
        try:
            self.go_to('approach', top, slow=False)
            for gap in a.gaps:
                self.stop_at(pt, idx, gap, 'down')
            if a.bidirectional:
                for gap in reversed(a.gaps[:-1]):
                    self.stop_at(pt, idx, gap, 'up')
        except SkipPoint as e:
            print(f'  -> {e}')
        finally:
            if a.go:
                try:
                    self.go_to('lift', top, slow=True)
                except SkipPoint as e:
                    print(f'  !! lift failed: {e}')

    # ---------------------------------------------------------------- report
    def save(self):
        if not self.rows:
            return
        with open(self.a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        print(f'\nCSV: {self.a.out}')

    def summary(self):
        rows = [r for r in self.rows if r['err_mm'] is not None]
        if not rows:
            print('no measurements entered')
            return
        err = np.array([r['err_mm'] for r in rows])
        print('\n================ SUMMARY ================')
        for d in ('down', 'up'):
            e = np.array([r['err_mm'] for r in rows if r['direction'] == d])
            if len(e):
                print(f'{d:4s}: n={len(e):2d}  mean err {e.mean():+.1f} mm  '
                      f'std {e.std():.1f}  min {e.min():+.1f}  max {e.max():+.1f}')
        print('per point (mean err mm, model dz mm, joint err deg):')
        for i in sorted({r['point'] for r in rows}):
            rr = [r for r in rows if r['point'] == i]
            print(f'  pt {i} ({rr[0]["x"] * 1000:.0f},{rr[0]["y"] * 1000:.0f}): '
                  f'{np.mean([r["err_mm"] for r in rr]):+6.1f} | '
                  f'{np.mean([r["model_dz_mm"] for r in rr]):+5.1f} | '
                  f'{np.mean([r["joint_err_max_deg"] for r in rr]):.2f}')
        if len({(r['x'], r['y']) for r in rows}) >= 3:
            A = np.c_[np.ones(len(rows)), [r['x'] * 1000 for r in rows],
                      [r['y'] * 1000 for r in rows]]
            c, *_ = np.linalg.lstsq(A, err, rcond=None)
            print(f'trend: err = {c[0]:+.1f} {c[1]:+.3f}*x[mm] {c[2]:+.3f}*y[mm]  '
                  '(slopes ~0 -> constant offset; large slopes -> kinematic/hand-eye error)')
        jm = np.mean([r['joint_err_max_deg'] for r in rows])
        sd = err.std()
        mean = err.mean()
        print(f'\nmean joint error {jm:.2f} deg, model dz '
              f'{np.mean([r["model_dz_mm"] for r in rows]):+.1f} mm')
        print(f'suggested --table-z (cancels the mean): {self.a.table_z - mean / 1000:.4f} m '
              f'(now {self.a.table_z:.4f})')
        worst = err.min()
        print(f'worst case tip too low by {-worst:.1f} mm -> keep a reserve of at least '
              f'{max(0.0, -worst + 2 * sd):.1f} mm (=min(err)-2 std) in --table-margin / gaps')
        if jm < 0.7 and sd + abs(mean) > 2.0:
            print('=> joints track fine but the tip does not: model / table-z error, '
                  'NOT the driver (PID would not help).')
        elif jm >= 0.7:
            print('=> joint tracking error is large: look at the driver (ff_bias/ff_dir, '
                  'settle correction).')
        else:
            print('=> error small: OK.')


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--points', default=DEFAULT_POINTS, help='"x,y;x,y;..." desk points in base [m]')
    a.add_argument('--gaps', default='30,20,12,6', help='descent stops, finger tip above the '
                   'desk model [mm], high -> low')
    a.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    a.add_argument('--approach', type=float, default=0.06, help='height above desk between points [m]')
    a.add_argument('--min-clearance', type=float, default=2.0,
                   help='stop descending at a point when the measured gap [mm] is below this')
    a.add_argument('--yaw', type=float, default=None, help='TCP yaw [rad] (default: current)')
    a.add_argument('--bidirectional', action='store_true',
                   help='after the lowest stop go back up through the stops and measure again')
    a.add_argument('--velocity', type=float, default=0.6)
    a.add_argument('--slow-velocity', type=float, default=0.15)
    a.add_argument('--accel', type=float, default=0.5)
    a.add_argument('--settle', type=float, default=1.0, help='wait before reading [s]')
    a.add_argument('--out', default=datetime.datetime.now().strftime('z_clearance_%Y%m%d_%H%M%S.csv'))
    a.add_argument('--go', action='store_true', help='really move (else plan only)')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    args, _ = a.parse_known_args()
    args.gaps = sorted([float(v) / 1000 for v in args.gaps.split(',')], reverse=True)
    if min(args.gaps) < 0.004:
        raise SystemExit('lowest gap must be >= 4 mm')
    points = parse_points(args.points)

    rclpy.init()
    node = Probe()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    t = None
    try:
        t = Test(node, args)
        print(f'driver connection: {t.conn}  go={args.go}  table_z={args.table_z}  '
              f'gaps(mm)={[g * 1000 for g in args.gaps]}  yaw={np.degrees(t.yaw):.0f} deg')
        if args.go:
            print('Keep your hand clear. Put a ruler / feeler gauge next to each point.')
        for i, pt in enumerate(points, 1):
            t.run_point(i, pt)
    except (Quit, KeyboardInterrupt):
        print('\nquit')
        if t is not None:
            t.mv.cancel()
    finally:
        if t is not None:
            t.save()
            t.summary()
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
