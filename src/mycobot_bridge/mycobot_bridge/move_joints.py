"""Send one joint goal (degrees, URDF convention) to the driver and wait.

  ros2 run mycobot_bridge move_joints 0 0 0 0 0 0
  ros2 run mycobot_bridge move_joints 0 -30 60 -30 0 0 --gripper 50
"""
import argparse
import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint

from mycobot_bridge.driver_node import ARM_JOINTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('deg', type=float, nargs='*', help='6 joint angles [deg]')
    ap.add_argument('--gripper', type=float, help='0 (closed) .. 100 (open)')
    ap.add_argument('--gripper-open-rad', type=float, default=1.0)
    a, ros_args = ap.parse_known_args()
    if a.deg and len(a.deg) != 6:
        ap.error('give exactly 6 angles')

    rclpy.init(args=ros_args)
    node = Node('move_joints')
    if a.deg:
        cli = ActionClient(node, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        cli.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[math.radians(d) for d in a.deg], time_from_start=Duration(sec=1))]
        res = _run(node, cli, goal)
        node.get_logger().info(f'arm: error_code={res.error_code} {res.error_string}')
    if a.gripper is not None:
        cli = ActionClient(node, GripperCommand, 'gripper_controller/gripper_cmd')
        cli.wait_for_server()
        goal = GripperCommand.Goal()
        goal.command.position = a.gripper / 100.0 * a.gripper_open_rad
        res = _run(node, cli, goal)
        node.get_logger().info(f'gripper: position={res.position:.3f} rad '
                               f'reached={res.reached_goal}')
    node.destroy_node()
    rclpy.shutdown()


def _run(node, cli, goal):
    f = cli.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, f)
    gh = f.result()
    if not gh.accepted:
        raise SystemExit('goal rejected (see driver log)')
    r = gh.get_result_async()
    rclpy.spin_until_future_complete(node, r)
    return r.result().result


if __name__ == '__main__':
    main()
