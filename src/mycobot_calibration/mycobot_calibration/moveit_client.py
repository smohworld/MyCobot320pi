"""Thin blocking wrappers around move_group's ROS interfaces.

  /compute_ik            IK with collision checking (joint limits incl. the J6 window
                         come from joint_limits.yaml through the robot model)
  /move_action           plan (OMPL) + execute a joint-space goal, collision free
  /apply_planning_scene  table box so planned motions stay above the table

Call these from a worker thread (not from an executor callback), the node must spin in
a MultiThreadedExecutor.
"""
import threading

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (CollisionObject, Constraints, JointConstraint, MoveItErrorCodes,
                             PlanningScene, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from mycobot_calibration.geometry import fill_pose_msg

ERR = {v: k for k, v in MoveItErrorCodes.__dict__.items()
       if k.isupper() and isinstance(v, int)}


def _wait(future, timeout):
    ev = threading.Event()
    future.add_done_callback(lambda _: ev.set())
    if not ev.wait(timeout):
        return None
    return future.result()


class MoveItClient:

    def __init__(self, node, group='arm', base_frame='base'):
        self.node = node
        self.group = group
        self.base = base_frame
        cb = ReentrantCallbackGroup()
        self.ik = node.create_client(GetPositionIK, 'compute_ik', callback_group=cb)
        self.scene = node.create_client(ApplyPlanningScene, 'apply_planning_scene',
                                        callback_group=cb)
        self.move = ActionClient(node, MoveGroup, 'move_action', callback_group=cb)

    def wait_ready(self, timeout=10.0):
        return (self.ik.wait_for_service(timeout_sec=timeout)
                and self.scene.wait_for_service(timeout_sec=timeout)
                and self.move.wait_for_server(timeout_sec=timeout))

    # ----------------------------------------------------------------- IK
    def compute_ik(self, T_base_link, link, seed_names, seed_q, timeout=0.2,
                   avoid_collisions=True):
        """Return joint positions (list, order of seed_names) or None."""
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.group
        r.ik_link_name = link
        r.avoid_collisions = avoid_collisions
        r.robot_state = RobotState(joint_state=JointState(
            name=list(seed_names), position=[float(v) for v in seed_q]))
        r.pose_stamped = PoseStamped()
        r.pose_stamped.header.frame_id = self.base
        fill_pose_msg(r.pose_stamped.pose, T_base_link)
        r.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        res = _wait(self.ik.call_async(req), timeout + 5.0)
        if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
            return None
        js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        if not all(n in js for n in seed_names):
            return None
        return [js[n] for n in seed_names]

    # -------------------------------------------------------------- motion
    def move_joints(self, names, q, velocity_scaling=0.3, accel_scaling=0.3,
                    planning_time=5.0, attempts=3, timeout=120.0):
        """Plan + execute. Returns (ok, message)."""
        g = MoveGroup.Goal()
        r = g.request
        r.group_name = self.group
        r.num_planning_attempts = int(attempts)
        r.allowed_planning_time = float(planning_time)
        r.max_velocity_scaling_factor = float(velocity_scaling)
        r.max_acceleration_scaling_factor = float(accel_scaling)
        r.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=n, position=float(v), tolerance_above=0.005,
                            tolerance_below=0.005, weight=1.0) for n, v in zip(names, q)])]
        g.planning_options.plan_only = False
        g.planning_options.replan = True
        g.planning_options.replan_attempts = 2
        gh = _wait(self.move.send_goal_async(g), 10.0)
        if gh is None or not gh.accepted:
            return False, 'move_group rejected the goal'
        self._active = gh
        res = _wait(gh.get_result_async(), timeout)
        self._active = None
        if res is None:
            gh.cancel_goal_async()
            return False, 'motion timed out'
        code = res.result.error_code.val
        return code == MoveItErrorCodes.SUCCESS, ERR.get(code, str(code))

    def cancel(self):
        gh = getattr(self, '_active', None)
        if gh is not None:
            gh.cancel_goal_async()

    # --------------------------------------------------------------- scene
    def add_box(self, name, center_xyz, size_xyz):
        co = CollisionObject()
        co.header.frame_id = self.base
        co.id = name
        co.operation = CollisionObject.ADD
        co.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                        dimensions=[float(v) for v in size_xyz])]
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in center_xyz)
        p.orientation.w = 1.0
        co.primitive_poses = [p]
        req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
        req.scene.world.collision_objects = [co]
        res = _wait(self.scene.call_async(req), 5.0)
        return res is not None and res.success

    def remove_object(self, name):
        co = CollisionObject()
        co.header.frame_id = self.base
        co.id = name
        co.operation = CollisionObject.REMOVE
        req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
        req.scene.world.collision_objects = [co]
        res = _wait(self.scene.call_async(req), 5.0)
        return res is not None and res.success
