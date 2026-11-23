import math, threading, numpy as np, rclpy
from rclpy.executors import MultiThreadedExecutor
from scipy.optimize import least_squares
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from moveit_msgs.msg import PlanningScene, RobotState
from sensor_msgs.msg import JointState
from mycobot_calibration.moveit_client import _wait
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.geometry import rot_z
from mycobot_tasks import layout as lay, place_scene as ps
from fastik import arm, LO, HI, DOWN
rclpy.init(); n = rclpy.create_node('shelfscan'); ex = MultiThreadedExecutor(); ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
ap = n.create_client(ApplyPlanningScene, 'apply_planning_scene'); sv = n.create_client(GetStateValidity, 'check_state_validity')
import os
if not (ap.wait_for_service(10) and sv.wait_for_service(10)):
    raise SystemExit(f"move_group not found (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '')!r} here): "
                     "start the sim launch in a terminal with the same ROS_DOMAIN_ID (echo $ROS_DOMAIN_ID)")
L = lay.Layout('~/.ros/layout.yaml'); orig = dict(L.v)
def apply_shelf(margin=0.01):
    a = L.args(); a.shelf_margin = margin
    o, _ = ps.build(a)
    req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
    req.scene.world.collision_objects = ps.collision_objects({k: v for k, v in o.items() if k.startswith('shelf')}, 'base')
    r = _wait(ap.call_async(req), 20.0)
    if r is None or not r.success: raise SystemExit('apply_planning_scene failed')
    return True, o
def valid(q):
    rq = GetStateValidity.Request(); rq.group_name = 'arm'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q]))
    r = _wait(sv.call_async(rq), 20.0)
    return all(set((c.contact_body_1, c.contact_body_2)) == {'base', 'desk'} for c in r.contacts)
rng = np.random.default_rng(2)
def good(p, b):
    R = rot_z(math.pi / 2) @ DOWN
    f = lambda q: np.r_[(arm.fk(q)[:3, 3] - p) * 10, (arm.fk(q)[:3, :3].T @ R - np.eye(3)).ravel() * 0.3]
    for _ in range(14):
        q0 = rng.uniform(LO, HI); q0[0] = b + rng.normal(0, 0.4)
        r = least_squares(f, np.clip(q0, LO, HI), bounds=(LO, HI))
        if np.linalg.norm(r.fun) < 3e-3 and abs(math.remainder(r.x[0] - b, 2 * math.pi)) <= math.radians(60) and valid(r.x):
            return True
    return False
