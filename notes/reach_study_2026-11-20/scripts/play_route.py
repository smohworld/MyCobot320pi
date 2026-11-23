# Play route_a.pkl on the SIM arm (FollowJointTrajectory + gripper_cmd), tray shown in RViz.
#   python3 play_route.py [--speed 0.35] [--pause]
import argparse, math, pickle, time, threading, numpy as np, rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from moveit_msgs.msg import (PlanningScene, CollisionObject, AttachedCollisionObject, RobotState,
                             Constraints, JointConstraint)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from fastik import arm
ARM = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3', 'joint5_to_joint4',
       'joint6_to_joint5', 'joint6output_to_joint6']
TOUCH = ['gripper_connection', 'gripper_base', 'gripper_left1', 'gripper_left2', 'gripper_left3',
         'gripper_right1', 'gripper_right2', 'gripper_right3', 'gripper_tcp']
ap_ = argparse.ArgumentParser(); ap_.add_argument('--speed', type=float, default=0.35, help='max joint speed [rad/s]')
ap_.add_argument('--pause', action='store_true', help='wait for Enter before each segment')
ap_.add_argument('--file', default='route_a.pkl'); A = ap_.parse_args()
R = pickle.load(open(A.file, 'rb')); DIMS = R['tray_dims']

rclpy.init(); n = rclpy.create_node('play_route'); ex = MultiThreadedExecutor(); ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
fjt = ActionClient(n, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
grp = ActionClient(n, GripperCommand, 'gripper_controller/gripper_cmd')
aps = n.create_client(ApplyPlanningScene, 'apply_planning_scene')
plan = n.create_client(GetMotionPlan, 'plan_kinematic_path')
js = {}; n.create_subscription(JointState, 'joint_states', lambda m: js.update(zip(m.name, m.position)), 10)
import os
if not all(c.wait_for_server(timeout_sec=10) for c in (fjt, grp)):
    raise SystemExit(f"sim arm not found (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '')!r} here): "
                     "start the sim launch in a terminal with the same ROS_DOMAIN_ID (echo $ROS_DOMAIN_ID)")
aps.wait_for_service(10); plan.wait_for_service(10); time.sleep(0.5)

def wait(fut, t=120.0):
    t0 = time.time()
    while not fut.done():
        if time.time() - t0 > t: raise TimeoutError
        time.sleep(0.01)
    return fut.result()
def action(cli, goal, t=300.0):
    h = wait(cli.send_goal_async(goal))
    if not h.accepted: raise SystemExit('goal rejected')
    return wait(h.get_result_async(), t).result
def apply(world=(), attached=()):
    rq = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
    rq.scene.robot_state.is_diff = True
    rq.scene.world.collision_objects = list(world); rq.scene.robot_state.attached_collision_objects = list(attached)
    wait(aps.call_async(rq))
def box(name, frame, c, quat, op=CollisionObject.ADD):
    o = CollisionObject(); o.id = name; o.header.frame_id = frame; o.operation = op
    if op == CollisionObject.ADD:
        o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(DIMS))]
        p = Pose(); p.position.x, p.position.y, p.position.z = map(float, c)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, quat); o.primitive_poses = [p]
    return o
def q_now(): return np.array([js[j] for j in ARM])
def run(qs, label):
    q0 = q_now(); jt = JointTrajectory(joint_names=ARM); t = 0.0
    pts = [q0] + list(qs)
    for a, b in zip(pts, pts[1:]):
        t += max(0.05, float(np.max(np.abs(b - a))) / A.speed)
        jt.points.append(JointTrajectoryPoint(positions=[float(v) for v in b],
                                              time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))
    g = FollowJointTrajectory.Goal(trajectory=jt)
    r = action(fjt, g)
    print(f'  {label}: {len(qs)} pts, {t:.1f} s -> {"ok" if r.error_code == 0 else r.error_string}', flush=True)
def ompl_to(q):
    rq = GetMotionPlan.Request(); m = rq.motion_plan_request
    m.start_state = RobotState(joint_state=JointState(name=list(js), position=list(js.values())))
    m.group_name = 'arm'; m.num_planning_attempts = 5; m.allowed_planning_time = 10.0
    m.max_velocity_scaling_factor = 0.3; m.max_acceleration_scaling_factor = 0.3
    m.goal_constraints = [Constraints(joint_constraints=[JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005,
                                                                         tolerance_below=0.005, weight=1.0) for j, v in zip(ARM, q)])]
    for k in range(8):
        res = wait(plan.call_async(rq)).motion_plan_response
        if res.error_code.val == 1: break
        print(f'  plan to the route start: try {k} failed ({res.error_code.val})')
    else: raise SystemExit('no plan to the route start')
    jt = res.trajectory.joint_trajectory; idx = [list(jt.joint_names).index(j) for j in ARM]
    return [np.array(p.positions)[idx] for p in jt.points]

name = 'route_tray'
apply(world=[box(name, 'base', [0, 0, 0], [0, 0, 0, 1], CollisionObject.REMOVE)])
first = next(s for k, s, _ in R['steps'] if k == 'traj')[0]
print(f"route A, desk spot {R['spot']}; speed {A.speed} rad/s")
action(grp, GripperCommand.Goal(command=type(GripperCommand.Goal().command)(position=0.5, max_effort=0.0)), 30)
run(ompl_to(first), 'from the current pose to the route start (OMPL)')
for k, data, label in R['steps']:
    if A.pause: input(f'[Enter] {label} ')
    if k == 'traj': run(data, label)
    elif k == 'grip':
        action(grp, GripperCommand.Goal(command=type(GripperCommand.Goal().command)(position=float(data), max_effort=0.0)), 30)
        print(f'  gripper {label}', flush=True)
    elif k == 'scene':
        op = data[0]
        if op == 'add':
            c, quat = data[2]; apply(world=[box(name, 'base', c, quat)])
        elif op == 'attach':
            a = AttachedCollisionObject(link_name='gripper_tcp', touch_links=TOUCH,
                                        object=box(name, 'gripper_tcp', [0, 0, 0.028], [0, 0, 0, 1]))
            apply(world=[box(name, 'base', 0, 0, CollisionObject.REMOVE)], attached=[a])
        elif op == 'detach':
            T = arm.fk(q_now()); c = T[:3, 3] + T[:3, 2] * 0.028; quat = Rotation.from_matrix(T[:3, :3]).as_quat()
            apply(attached=[AttachedCollisionObject(link_name='gripper_tcp', object=box(name, 'gripper_tcp', 0, 0, CollisionObject.REMOVE))])
            apply(world=[box(name, 'base', c, quat)])
        print(f'  scene: {label}', flush=True)
print('done (tray box left in the scene as "route_tray"; rerun removes it)')
