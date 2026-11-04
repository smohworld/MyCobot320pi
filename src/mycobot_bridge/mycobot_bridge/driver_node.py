"""ROS 2 driver for myCobot 320 (+ myGripper F100) using pymycobot.

Publishes:
  /joint_states                     sensor_msgs/JointState (arm + gripper_controller)
Actions:
  arm_controller/follow_joint_trajectory   control_msgs/FollowJointTrajectory
      each trajectory point is sent with send_angles(); intermediate points are
      passed when within waypoint_tolerance_deg, the last one must settle within
      goal_tolerance_deg (same interface MoveIt uses later).
  gripper_controller/gripper_cmd            control_msgs/GripperCommand
      position = gripper_controller joint value [rad], 0 closed .. gripper_joint_open
Services:
  release_servos  std_srvs/SetBool   true: servos off (hand teaching), false: servos on
  stop            std_srvs/Trigger   stop the arm immediately

Connection (parameter `connection`):
  socket  PC -> robot_side/mycobot320_socket_server.py running on the robot's Pi
  serial  USB cable (or /dev/ttyAMA0 when this node runs on the Pi itself)
  fake    simulated robot, no hardware
"""
import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory, GripperCommand
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

from mycobot_bridge.robot_interface import MyCobotInterface

ARM_JOINTS = [
    'joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
    'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6',
]
# from the URDF limits [rad] -> deg
DEFAULT_LIMITS_DEG = [167.8, 134.6, 144.9, 144.9, 167.8, 179.9]


class MyCobotDriver(Node):

    def __init__(self):
        super().__init__('mycobot_driver')
        p = self.declare_parameter
        p('connection', 'socket')              # socket | serial | fake
        p('ip', '192.168.0.10')
        p('netport', 9000)
        p('serial_port', '/dev/ttyUSB0')
        p('baudrate', 115200)
        p('joint_names', ARM_JOINTS)
        # urdf_deg = sign * robot_deg + offset  (identity unless RViz and the real arm disagree)
        p('joint_signs', [1.0] * 6)
        p('joint_offsets_deg', [0.0] * 6)
        p('joint_limits_deg', DEFAULT_LIMITS_DEG)
        p('state_rate', 10.0)
        p('move_speed', 30)                    # pymycobot speed 1..100
        p('goal_tolerance_deg', 1.0)
        p('waypoint_tolerance_deg', 5.0)
        p('settle_time', 0.5)                  # extra wait after the last point [s]
        p('move_timeout', 30.0)                # per trajectory point [s]
        p('power_on_at_start', True)
        p('gripper_enabled', True)
        p('gripper_id', 14)
        p('gripper_joint', 'gripper_controller')
        p('gripper_joint_open', 1.0)           # URDF joint value [rad] at gripper 100 (open)
        p('gripper_rate', 2.0)
        p('gripper_timeout', 3.0)

        g = self.get_parameter
        self.joint_names = list(g('joint_names').value)
        self.signs = list(g('joint_signs').value)
        self.offsets = list(g('joint_offsets_deg').value)
        self.limits = list(g('joint_limits_deg').value)
        self.speed = int(g('move_speed').value)
        self.goal_tol = float(g('goal_tolerance_deg').value)
        self.wp_tol = float(g('waypoint_tolerance_deg').value)
        self.settle = float(g('settle_time').value)
        self.move_timeout = float(g('move_timeout').value)
        self.gripper_enabled = bool(g('gripper_enabled').value)
        self.gripper_joint = g('gripper_joint').value
        self.gripper_open = float(g('gripper_joint_open').value)
        self.gripper_timeout = float(g('gripper_timeout').value)

        conn = g('connection').value
        self.get_logger().info(f'connecting ({conn}) ...')
        self.robot = MyCobotInterface(
            conn, ip=g('ip').value, netport=int(g('netport').value),
            serial_port=g('serial_port').value, baudrate=int(g('baudrate').value),
            gripper_id=int(g('gripper_id').value))
        if g('power_on_at_start').value:
            self.robot.power_on()
            time.sleep(0.5)

        self._state_lock = threading.Lock()
        self._robot_deg = None       # last valid reading, robot units
        self._prev_robot_deg = None
        self._gripper_val = None     # 0..100
        self._busy = threading.Lock()
        self._bad_reads = 0

        self.js_pub = self.create_publisher(JointState, 'joint_states', 10)
        state_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(1.0 / float(g('state_rate').value), self._poll_state,
                          callback_group=state_group)
        if self.gripper_enabled:
            self.create_timer(1.0 / float(g('gripper_rate').value), self._poll_gripper,
                              callback_group=state_group)

        cb = ReentrantCallbackGroup()
        self.arm_server = ActionServer(
            self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_trajectory, goal_callback=self._goal_cb,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        if self.gripper_enabled:
            self.gripper_server = ActionServer(
                self, GripperCommand, 'gripper_controller/gripper_cmd',
                execute_callback=self._execute_gripper,
                cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.create_service(SetBool, 'release_servos', self._release_cb, callback_group=cb)
        self.create_service(Trigger, 'stop', self._stop_cb, callback_group=cb)

        first = self.robot.get_angles()
        if first is None:
            self.get_logger().error('no valid get_angles() reply - check connection / power')
        else:
            self.get_logger().info(f'connected, joints [deg] = {first}')

    # --- unit conversion -------------------------------------------------
    def robot_to_urdf_rad(self, robot_deg):
        return [math.radians(s * d + o) for d, s, o in zip(robot_deg, self.signs, self.offsets)]

    def urdf_rad_to_robot(self, rad):
        return [(math.degrees(r) - o) / s for r, s, o in zip(rad, self.signs, self.offsets)]

    # --- state -----------------------------------------------------------
    def _poll_state(self):
        a = self.robot.get_angles()
        if a is None:
            self._bad_reads += 1
            if self._bad_reads in (5, 50) or self._bad_reads % 500 == 0:
                self.get_logger().warn(f'{self._bad_reads} consecutive bad get_angles() replies')
            return
        self._bad_reads = 0
        with self._state_lock:
            self._prev_robot_deg, self._robot_deg = self._robot_deg, a
            grip = self._gripper_val
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = self.robot_to_urdf_rad(a)
        if self.gripper_enabled and grip is not None:
            msg.name.append(self.gripper_joint)
            msg.position.append(grip / 100.0 * self.gripper_open)
        self.js_pub.publish(msg)

    def _poll_gripper(self):
        v = self.robot.get_gripper()
        if v is not None:
            with self._state_lock:
                self._gripper_val = v

    def _latest(self):
        with self._state_lock:
            return self._robot_deg, self._prev_robot_deg

    # --- arm trajectory --------------------------------------------------
    def _goal_cb(self, goal):
        names = list(goal.trajectory.joint_names)
        if sorted(names) != sorted(self.joint_names):
            self.get_logger().error(f'trajectory joints {names} != {self.joint_names}')
            return GoalResponse.REJECT
        if not goal.trajectory.points:
            return GoalResponse.REJECT
        if self._busy.locked():
            self.get_logger().error('already executing a trajectory')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_trajectory(self, goal_handle):
        result = FollowJointTrajectory.Result()
        traj = goal_handle.request.trajectory
        idx = [list(traj.joint_names).index(n) for n in self.joint_names]
        targets = [self.urdf_rad_to_robot([pt.positions[i] for i in idx]) for pt in traj.points]

        for t in targets:
            bad = [j for j, (v, lim) in enumerate(zip(t, self.limits)) if abs(v) > lim]
            if bad:
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = f'joint(s) {bad} out of limits: {[round(v, 1) for v in t]}'
                self.get_logger().error(result.error_string)
                goal_handle.abort()
                return result

        with self._busy:
            fb = FollowJointTrajectory.Feedback()
            fb.joint_names = list(self.joint_names)
            for k, target in enumerate(targets):
                last = k == len(targets) - 1
                self.robot.send_angles(target, self.speed)
                ok = self._wait_reached(target, self.goal_tol if last else self.wp_tol,
                                        need_still=last, goal_handle=goal_handle, fb=fb)
                if ok == 'canceled':
                    self.robot.stop()
                    goal_handle.canceled()
                    result.error_string = 'canceled'
                    return result
                if not ok:
                    self.robot.stop()
                    cur, _ = self._latest()
                    result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                    result.error_string = (f'point {k} not reached in {self.move_timeout}s; '
                                           f'target {[round(v, 1) for v in target]} '
                                           f'actual {cur}')
                    self.get_logger().error(result.error_string)
                    goal_handle.abort()
                    return result
            time.sleep(self.settle)

        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        return result

    def _wait_reached(self, target, tol, need_still, goal_handle, fb):
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.move_timeout:
            if goal_handle.is_cancel_requested:
                return 'canceled'
            cur, prev = self._latest()
            if cur is not None:
                err = max(abs(c - t) for c, t in zip(cur, target))
                still = prev is not None and max(abs(c - p) for c, p in zip(cur, prev)) < 0.2
                fb.actual.positions = self.robot_to_urdf_rad(cur)
                fb.desired.positions = self.robot_to_urdf_rad(target)
                fb.error.positions = [
                    a - d for a, d in zip(fb.actual.positions, fb.desired.positions)]
                goal_handle.publish_feedback(fb)
                if err < tol and (still or not need_still):
                    return True
            time.sleep(0.05)
        return False

    # --- gripper ---------------------------------------------------------
    def _execute_gripper(self, goal_handle):
        cmd = goal_handle.request.command
        value = max(0.0, min(100.0, cmd.position / self.gripper_open * 100.0))
        self.robot.set_gripper(value)
        result = GripperCommand.Result()
        t0, last, stable = time.monotonic(), None, 0
        while time.monotonic() - t0 < self.gripper_timeout:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result
            time.sleep(0.2)
            v = self.robot.get_gripper()
            if v is None:
                continue
            with self._state_lock:
                self._gripper_val = v
            stable = stable + 1 if last is not None and abs(v - last) < 1.0 else 0
            last = v
            result.reached_goal = abs(v - value) < 3.0
            if result.reached_goal or stable >= 3:   # reached, or blocked by an object
                break
        if last is not None:
            result.position = last / 100.0 * self.gripper_open
        result.stalled = not result.reached_goal
        goal_handle.succeed()
        return result

    # --- services --------------------------------------------------------
    def _release_cb(self, req, res):
        if self._busy.locked():
            res.success, res.message = False, 'trajectory running'
            return res
        if req.data:
            self.robot.release_all_servos()
            res.message = 'servos released - support the arm, it will drop'
        else:
            self.robot.focus_all_servos()
            res.message = 'servos locked'
        res.success = True
        return res

    def _stop_cb(self, req, res):
        self.robot.stop()
        res.success, res.message = True, 'stopped'
        return res

    def destroy_node(self):
        self.robot.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MyCobotDriver()
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
