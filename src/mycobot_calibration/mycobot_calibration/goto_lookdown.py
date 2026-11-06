"""Move the arm so the camera looks straight down at the board centre: the start pose for
the hand-eye ~/plan (max_joint_dev_deg keeps all viewpoints near this arm configuration).

Needs move_group running (e.g. handeye.launch.py). Lists the IK candidates only, unless --go:

  ros2 run mycobot_calibration goto_lookdown 0.31 0.0                  # list candidates
  ros2 run mycobot_calibration goto_lookdown 0.31 0.0 --go             # move (fake robot)
  ros2 run mycobot_calibration goto_lookdown 0.31 0.0 --go --real      # move the REAL robot

x y = board centre in the base frame [m]. --height is tried first; if the camera cannot get
that high above the board, lower heights are tried (down to --min-height). Solutions must keep
J6 inside the +-90 deg cable window; the one farthest from the joint limits is used
(--pick N for another one from the list).
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.geometry import inv, look_at, make_T, rot_z, transform_msg_to_T
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient

# joint limits J1..J6 [deg] (URDF; J6: camera cable window)
LIMITS = np.radians([167.9, 134.6, 145.0, 145.0, 167.9, 90.0])
SEEDS_J23 = [(0.0, 0.0), (-0.6, 1.2), (0.6, -1.2)]


def driver_connection(node):
    cli = node.create_client(GetParameters, '/mycobot_driver/get_parameters')
    if not cli.wait_for_service(timeout_sec=5.0):
        return None
    fut = cli.call_async(GetParameters.Request(names=['connection']))
    ev = threading.Event()
    fut.add_done_callback(lambda _: ev.set())
    if not ev.wait(5.0) or not fut.result().values:
        return None
    return fut.result().values[0].string_value


def lookdown_solutions(mv, X, T_l6_tcp, target, height, min_tcp_z=-np.inf):
    """IK solutions (margin, roll_deg, q) for the camera at `height` above target.
    Poses that put the gripper TCP (finger tip) below min_tcp_z are skipped."""
    T0 = look_at(target + [0.0, 0.0, height], target)
    sols = []
    for roll in np.radians(np.arange(0, 360, 45)):
        T_ik = T0 @ make_T(rot_z(roll), np.zeros(3)) @ inv(X) @ T_l6_tcp
        if T_ik[2, 3] < min_tcp_z:
            continue
        for j1 in np.radians(np.arange(-150, 151, 30)):
            for j2, j3 in SEEDS_J23:
                q = mv.compute_ik(T_ik, 'gripper_tcp', ARM_JOINTS,
                                  [j1, j2, j3, 0.0, 0.0, 0.0], timeout=0.05)
                if q is None:
                    continue
                q = np.array(q)
                margin = float(np.min(1 - np.abs(q) / LIMITS))
                if margin > 0:
                    sols.append((margin, float(np.degrees(roll)), q))
    uniq = []
    for s in sorted(sols, key=lambda s: -s[0]):
        if all(np.max(np.abs(s[2] - u[2])) > np.radians(5) for u in uniq):
            uniq.append(s)
    return uniq


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('x', type=float, help='board centre x in base [m]')
    a.add_argument('y', type=float, help='board centre y in base [m]')
    a.add_argument('--z', type=float, default=-0.007, help='board top z in base [m]')
    a.add_argument('--height', type=float, default=0.28, help='camera above the board [m]')
    a.add_argument('--min-height', type=float, default=0.16)
    a.add_argument('--min-tcp-clearance', type=float, default=0.025,
                   help='finger tip (gripper_tcp) at least this far above the board [m]')
    a.add_argument('--table-margin', type=float, default=0.012,
                   help='collision table top = z + this (same as handeye table_margin) [m]')
    a.add_argument('--go', action='store_true', help='move to the chosen solution')
    a.add_argument('--pick', type=int, default=0, help='index in the candidate list')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    a.add_argument('--velocity', type=float, default=0.2, help='velocity/accel scaling')
    args, _ = a.parse_known_args()             # ignore --ros-args from ros2 run

    rclpy.init()
    node = Node('goto_lookdown')
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        run(node, args)
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


def run(node, args):
    conn = driver_connection(node)
    if args.go and conn != 'fake' and not args.real:
        raise SystemExit(f'driver connection is {conn!r}, not fake: add --real to move it')

    buf = Buffer()
    TransformListener(buf, node)
    js = {}
    node.create_subscription(JointState, 'joint_states',
                             lambda m: js.update(zip(m.name, m.position)), 10)
    mv = MoveItClient(node)
    if not mv.wait_ready(15.0):
        raise SystemExit('move_group not available (start handeye.launch.py first)')
    t0 = time.monotonic()
    X = T_l6_tcp = None
    while time.monotonic() - t0 < 5.0:
        try:
            now = rclpy.time.Time()
            X = transform_msg_to_T(
                buf.lookup_transform('link6', 'camera_optical_frame', now).transform)
            T_l6_tcp = transform_msg_to_T(
                buf.lookup_transform('link6', 'gripper_tcp', now).transform)
        except Exception:      # noqa: BLE001 - TF not there yet
            pass
        if X is not None and all(n in js for n in ARM_JOINTS):
            break
        time.sleep(0.2)
    else:
        raise SystemExit('no TF link6 -> camera_optical_frame / gripper_tcp or no joint_states')

    # desk in the planning scene (raised by the safety margin), so this move is collision
    # checked against it even before the calibrator's ~/plan added it
    if mv.add_box('calibration_table', [0.0, 0.0, args.z + args.table_margin - 0.025],
                  [1.5, 1.5, 0.05]):
        print(f'table in the planning scene: collision top z={args.z + args.table_margin:.3f} m')
    else:
        print('WARNING: could not add the table to the planning scene')

    target = np.array([args.x, args.y, args.z])
    min_tcp_z = args.z + args.min_tcp_clearance
    tcp_below = (T_l6_tcp[2, 3] - X[2, 3])     # finger tip is this far beyond the camera
    h, sols = args.height, []
    while h >= args.min_height - 1e-9:
        sols = lookdown_solutions(mv, X, T_l6_tcp, target, h, min_tcp_z)
        if sols:
            break
        why = ('finger tip too low' if h - tcp_below < args.min_tcp_clearance
               else 'unreachable')
        print(f'height {h:.2f} m: {why}')
        h -= 0.02
    if not sols:
        raise SystemExit('no IK solution with the finger tip >= '
                         f'{args.min_tcp_clearance * 1000:.0f} mm above the board: move the '
                         'board closer to the robot')

    print(f'camera {h:.2f} m above ({args.x:.3f}, {args.y:.3f}), finger tip ~'
          f'{(h - tcp_below) * 1000:.0f} mm above the board: {len(sols)} solutions, '
          'joints J1..J6 [deg], best first')
    for i, (m, r, q) in enumerate(sols[:8]):
        print(f'  [{i}] margin {m * 100:3.0f}%  roll {r:4.0f}  ' +
              ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
    if not args.go:
        print('add --go to move there' + ('' if conn == 'fake' else ' (and --real)'))
        return
    q = sols[min(args.pick, len(sols) - 1)][2]
    print(f'moving to [{min(args.pick, len(sols) - 1)}] ({conn}) ...')
    ok, msg = mv.move_joints(ARM_JOINTS, q, args.velocity, args.velocity)
    print('done' if ok else f'move failed: {msg}')


if __name__ == '__main__':
    main()
