"""Look at a point (x, y) on a shelf: camera (or gripper) facing it from the front.

A book on the desk stands in for a shelf (top --shelf-height over the desk). The aim point
is (x, y, desk + --shelf-height + --aim-dz). The view direction is horizontal, pointing from
the base axis out toward (x, y), or --yaw (perpendicular to a shelf face: "from the front"); --pitch tilts it down (oblique view
from above), e.g. --pitch 30. The finger tip stays --standoff in front of the aim point
along the view ray (the tip sticks out ~16 cm past the camera, so that sets how far back
the camera is).

Needs move_group and the driver running (e.g. handeye.launch.py). Lists only, unless --go:

  ros2 run mycobot_tasks look_toward 0.25 0.05                      # plan only
  ros2 run mycobot_tasks look_toward 0.25 0.05 --go --real          # horizontal, from the front
  ros2 run mycobot_tasks look_toward 0.25 0.05 --pitch 30 --go --real   # oblique, 30 deg down
  ros2 run mycobot_tasks look_toward 0.25 0.05 --aim tcp            # gripper axis at the point
  ros2 run mycobot_tasks look_toward 0.25 0.05 --book 0.30,0.05,0.15,0.22   # book as obstacle
  # standing book: face toward the robot at x=0.20, y -0.16 .. -0.455, top 21.5 cm
  ros2 run mycobot_tasks look_toward 0.20 -0.20 --yaw 0 --shelf-height 0.215 \\
      --book 0.215,-0.3075,0.03,0.295

If the exact pose is out of reach, nearby standoffs (+-2 cm steps, up to --search-standoff)
and, only with --pitch-search, nearby pitches are tried; what was used is printed.
Image kept upright first (camera roll 0), then other rolls; J6 stays in its +-90 deg window.
"""
import argparse
import itertools
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import inv, make_T, rot_z, transform_msg_to_T
from mycobot_calibration.goto_lookdown import LIMITS, SEEDS_J23, driver_connection
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient

CAM = 'camera_optical_frame'
TCP = 'gripper_tcp'
ROLLS_DEG = (0, 20, -20, 45, -45, 90, -90, 180)   # camera roll about the view axis, best first


def view_frame(eye, d):
    """Frame at eye with z = d (view direction), x = image right, y = image down (upright)."""
    z = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0])
    if abs(z @ up) > 0.99:                     # looking straight down: any yaw
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(-up, z)
    x /= np.linalg.norm(x)
    return make_T(np.column_stack([x, np.cross(z, x), z]), eye)


class LookNode(Node):
    def __init__(self):
        super().__init__('look_toward')
        self.js = {}
        self.create_subscription(JointState, 'joint_states',
                                 lambda m: self.js.update(zip(m.name, m.position)), 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)

    def T(self, parent, child):
        try:
            return transform_msg_to_T(
                self.tf.lookup_transform(parent, child, rclpy.time.Time()).transform)
        except Exception:  # noqa: BLE001
            return None

    def q(self):
        return np.array([self.js[n] for n in ARM_JOINTS]) if all(
            n in self.js for n in ARM_JOINTS) else None


def solve(mv, args, target, d, standoff, T_aim_tcp, tip_ahead, q_now, min_tcp_z):
    """Best IK (cost, q, roll_deg) for the aim frame looking along d at target, or None."""
    eye = target - d * (standoff + tip_ahead)     # aim frame origin (camera or tcp)
    T0 = view_frame(eye, d)
    j1 = float(np.arctan2(target[1], target[0]))
    best = None
    for roll in ROLLS_DEG:
        T_ik = T0 @ make_T(rot_z(np.radians(roll)), np.zeros(3)) @ T_aim_tcp
        if T_ik[2, 3] < min_tcp_z:
            continue
        for dj1, (j2, j3) in itertools.product(
                (0.0, -0.3, 0.3), SEEDS_J23 + [(0.3, -0.9), (-0.3, -0.6)]):
            q = mv.compute_ik(T_ik, TCP, ARM_JOINTS, [j1 + dj1, j2, j3, 0.0, 0.0, 0.0],
                              timeout=0.05)
            if q is None:
                continue
            q = np.asarray(q)
            if np.min(1 - np.abs(q) / LIMITS) <= 0.05:          # keep 5 % off the limits
                continue
            cost = (abs(roll) / 20.0) ** 2 + 5.0 * (q[0] - j1) ** 2 + \
                float(np.sum((q - q_now) ** 2))
            if best is None or cost < best[0]:
                best = (cost, q, roll)
        if best is not None and roll == 0:
            break                                  # upright image found: good enough
    return best


def run(node, args):
    conn = driver_connection(node)
    if args.go and conn != 'fake' and not args.real:
        raise SystemExit(f'driver connection is {conn!r}, not fake: add --real to move it')
    mv = MoveItClient(node)
    if not mv.wait_ready(15.0):
        raise SystemExit('move_group not available (start handeye.launch.py first)')
    t0 = time.monotonic()
    while (node.q() is None or node.T('link6', CAM) is None
           or node.T('link6', TCP) is None):
        if time.monotonic() - t0 > 5.0:
            raise SystemExit(f'no joint_states / TF link6 -> {CAM} / {TCP}')
        time.sleep(0.1)
    X = node.T('link6', CAM)
    T_l6_tcp = node.T('link6', TCP)
    q_now = node.q()

    # aim frame -> tcp, and how far the finger tip sticks out past the aim frame (along z)
    T_aim_tcp = inv(X) @ T_l6_tcp if args.aim == 'camera' else np.eye(4)
    tip_ahead = float(T_aim_tcp[2, 3])

    mv.add_box('calibration_table', [0.0, 0.0, args.table_z + args.table_margin - 0.025],
               [1.5, 1.5, 0.05])
    shelf_z = args.table_z + args.shelf_height
    if args.book:
        cx, cy, sx, sy = (float(v) for v in args.book.split(','))
        h = args.shelf_height + args.book_margin
        mv.add_box('book', [cx, cy, args.table_z + h / 2], [sx + 2 * args.book_margin,
                                                             sy + 2 * args.book_margin, h])
        print(f'book in the planning scene: centre ({cx * 1000:.0f}, {cy * 1000:.0f}) mm, '
              f'{sx * 1000:.0f} x {sy * 1000:.0f} mm, top {shelf_z * 1000:.0f} mm')
    else:
        mv.remove_object('book')
        print('no --book: the book is NOT a collision object (the arm may plan through it)')

    target = np.array([args.x, args.y, shelf_z + args.aim_dz])
    r = float(np.hypot(args.x, args.y))
    if r < 0.05 and args.yaw is None:
        raise SystemExit('(x, y) too close to the base axis: no "front" direction')
    az = np.arctan2(args.y, args.x) if args.yaw is None else np.radians(args.yaw)
    min_tcp_z = args.table_z + args.table_margin + 0.02

    pitches = [args.pitch]
    if args.pitch_search:
        for k in range(1, 4):
            pitches += [args.pitch + 10 * k, args.pitch - 10 * k]
        pitches = [p for p in pitches if -30 <= p <= 90]
    standoffs = [args.standoff]
    s = 0.02
    while s <= args.search_standoff + 1e-9:
        standoffs += [args.standoff + s, args.standoff - s]
        s += 0.02
    standoffs = [v for v in standoffs if v >= 0.0]
    cands = sorted(itertools.product(pitches, standoffs),
                   key=lambda c: abs(c[0] - args.pitch) / 10 + abs(c[1] - args.standoff) / 0.02)

    for pitch, standoff in cands:
        p = np.radians(pitch)
        d = np.array([np.cos(p) * np.cos(az), np.cos(p) * np.sin(az), -np.sin(p)])
        best = solve(mv, args, target, d, standoff, T_aim_tcp, tip_ahead, q_now, min_tcp_z)
        if best is not None:
            break
        print(f'  pitch {pitch:5.1f} deg, standoff {standoff * 1000:4.0f} mm: no IK')
    else:
        raise SystemExit('no reachable pose: move the point closer, lower --standoff, '
                         'try --pitch / --pitch-search or --aim tcp')

    _, q, roll = best
    print(f'aim point ({target[0] * 1000:.0f}, {target[1] * 1000:.0f}, '
          f'{target[2] * 1000:.0f}) mm, {args.aim} looking '
          f'{"horizontally" if abs(pitch) < 1e-6 else f"{pitch:.0f} deg down"} toward it '
          f'(azimuth {np.degrees(az):.1f} deg), finger tip {standoff * 1000:.0f} mm in front, '
          f'roll {roll} deg')
    print('  J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
    if not args.go:
        print('add --go to move there' + ('' if conn == 'fake' else ' (and --real)'))
        return
    print(f'moving ({conn}) ...')
    ok, msg = mv.move_joints(ARM_JOINTS, q, args.velocity, args.accel)
    print('done' if ok else f'move failed: {msg}')
    time.sleep(0.5)

    # check with the measured joints: where does the view axis pass the aim point
    frame = CAM if args.aim == 'camera' else TCP
    T = node.T('base', frame)
    if T is not None:
        o, z = T[:3, 3], T[:3, 2]
        v = target - o
        miss = np.linalg.norm(v - (v @ z) * z)
        print(f'measured: {frame} axis passes the aim point at {miss * 1000:.1f} mm, '
              f'{v @ z * 1000:.0f} mm ahead, pitch {np.degrees(np.arcsin(-z[2])):.1f} deg')


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('x', type=float, help='aim point x in base [m]')
    a.add_argument('y', type=float, help='aim point y in base [m]')
    a.add_argument('--shelf-height', type=float, default=0.21,
                   help='shelf (book) top over the desk [m]')
    a.add_argument('--aim-dz', type=float, default=0.0,
                   help='aim point = shelf top + this [m] (e.g. +0.03: something on the shelf, '
                        '-0.03: the front face of the book)')
    a.add_argument('--pitch', type=float, default=0.0,
                   help='view direction below horizontal [deg]; 0 = straight from the front '
                        '(horizontal), e.g. 30 = oblique from above')
    a.add_argument('--pitch-search', action='store_true',
                   help='also try pitches +-10/20/30 deg if --pitch is out of reach')
    a.add_argument('--yaw', type=float, default=None,
                   help='view direction azimuth in base [deg] (0 = +x, 90 = +y); default: '
                        'from the base axis out toward (x, y). For a shelf face use its '
                        'normal: a face at x = const looking at the robot -> --yaw 0')
    a.add_argument('--standoff', type=float, default=0.06,
                   help='finger tip distance in front of the aim point along the view [m]')
    a.add_argument('--search-standoff', type=float, default=0.06,
                   help='if out of reach, try standoffs up to +- this (2 cm steps) [m]')
    a.add_argument('--aim', choices=('camera', 'tcp'), default='camera',
                   help='which axis points at the aim point: camera optical axis or gripper')
    a.add_argument('--book', default='',
                   help='cx,cy,sx,sy [m]: book footprint (axis aligned, base frame) as a '
                        'collision box from the desk up to --shelf-height')
    a.add_argument('--book-margin', type=float, default=0.01,
                   help='collision box padding around the book [m]')
    a.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    a.add_argument('--table-margin', type=float, default=0.012,
                   help='desk collision box raised by this [m]')
    a.add_argument('--velocity', type=float, default=0.3, help='velocity scaling')
    a.add_argument('--accel', type=float, default=0.3, help='acceleration scaling')
    a.add_argument('--go', action='store_true', help='really move (else plan only)')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    args, _ = a.parse_known_args()

    rclpy.init()
    node = LookNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        run(node, args)
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
