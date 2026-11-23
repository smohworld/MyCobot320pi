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
ap.wait_for_service(5); sv.wait_for_service(5)
L = lay.Layout('~/.ros/layout.yaml'); orig = dict(L.v)
def apply_shelf():
    o, _ = ps.build(L.args())
    req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
    req.scene.world.collision_objects = ps.collision_objects({k: v for k, v in o.items() if k.startswith('shelf')}, 'base')
    return _wait(ap.call_async(req), 5.0).success, o
def valid(q):
    rq = GetStateValidity.Request(); rq.group_name = 'arm_group'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q]))
    r = _wait(sv.call_async(rq), 5.0)
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
try:
    for yaw in (90.0, 80.0, 100.0):
        for ox in np.arange(0.36, 0.57, 0.03):
            for oy in (0.05, 0.08, 0.11, 0.14):
                L.v.update(shelf=[float(ox), oy, yaw])
                ok, o = apply_shelf()
                if ps.box_hits(o, [k for k in o if k.startswith('shelf')], ['desk', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box']):
                    continue
                x, y, z = L.fork_point(); b = math.atan2(y, x); hz = z + 0.072
                g = [good(np.array([x, y, hz + h]), b) for h in (-0.02, 0.02, 0.04)]
                print(f'shelf {ox:.2f},{oy:.2f},{yaw:.0f}  D11 ({x:+.3f},{y:+.3f}) r {math.hypot(x, y)*100:.1f}  grasp/+2/+4: ' + ' '.join('Y' if v else '-' for v in g), flush=True)
                break   # smallest clear oy for this ox
finally:
    L.v.update(orig); apply_shelf(); print('scene restored')
