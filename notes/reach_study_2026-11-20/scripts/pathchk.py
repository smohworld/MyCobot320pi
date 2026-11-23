import math, numpy as np, rclpy
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from sensor_msgs.msg import JointState
from scanlib import n, valid, _wait, sv
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_tasks.red_cube_hover import DOWN, TCP
from moveit_msgs.srv import GetStateValidity
from fastik import solve
js = {}
n.create_subscription(JointState, 'joint_states', lambda m: js.update(zip(m.name, m.position)), 10)
import time; time.sleep(1.0)
q_now = np.array([js[j] for j in ARM_JOINTS]); print('now (deg)', np.round(np.degrees(q_now)).astype(int))
mv = MoveItClient(n, base_frame='base')
# reversed goal as layout would find (fingers across the handle: tray yaw 180 + 90)
p = np.array([0.003, 0.313, -0.043 + 0.072 + 0.04]); ty = math.pi
j1 = math.remainder(math.atan2(p[1], p[0]) + math.pi, 2 * math.pi)
goal = None
for k in (0, 1):
    T = make_T(rot_z(ty + math.pi / 2 + k * math.pi) @ DOWN, p)
    for a2, a3, a4 in ((-1.2, -0.6, 0.3), (0.8, 1.2, -0.4), (1.2, 0.6, -0.3), (-0.3, 1.8, 0.0)):
        q = mv.compute_ik(T, TCP, ARM_JOINTS, np.array([j1, a2, a3, a4, 0, 0]), timeout=0.1)
        if q is not None and abs(math.remainder(q[0] - j1, 2 * math.pi)) < math.radians(60): goal = np.array(q); break
    if goal is not None: break
print('goal (deg)', None if goal is None else np.round(np.degrees(goal)).astype(int))
def contacts(q):
    rq = GetStateValidity.Request(); rq.group_name = 'arm'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q]))
    r = _wait(sv.call_async(rq), 20.0)
    return sorted({(c.contact_body_1, c.contact_body_2) for c in r.contacts} - {('base', 'desk'), ('desk', 'base')})
cli = n.create_client(GetMotionPlan, 'plan_kinematic_path'); cli.wait_for_service(5)
print('current pose contacts:', contacts(q_now))
# bot-above (normal arm) start
pb = np.array([0.2155, -0.0449, -0.208 + 0.04]); jb = math.atan2(pb[1], pb[0]); qb = None
for a2, a3, a4 in ((0.0, -1.2, 0.0), (-0.8, -1.2, 0.4), (0.3, -1.8, 0.0), (0.6, -2.0, -0.2), (-1.2, -0.6, 0.3)):
    q = mv.compute_ik(make_T(rot_z(math.pi + math.pi / 2) @ DOWN, pb), TCP, ARM_JOINTS, np.array([jb, a2, a3, a4, 0, 0]), timeout=0.1)
    if q is not None and abs(math.remainder(q[0] - jb, 2 * math.pi)) < math.radians(60): qb = np.array(q); break
print('bot-above (deg)', None if qb is None else np.round(np.degrees(qb)).astype(int))
print('start pose contacts:', contacts(np.zeros(6)))
from scanlib import apply_shelf, L, orig
starts = [('start pose', np.zeros(6)), ('bot above', qb)]
for trial in range(8):
    apply_shelf(0.01)
    sname, q0 = starts[trial % len(starts)]
    rq = GetMotionPlan.Request(); r = rq.motion_plan_request
    r.start_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q0]))
    r.group_name = 'arm'; r.num_planning_attempts = 5; r.allowed_planning_time = 10.0
    r.max_velocity_scaling_factor = 0.3; r.max_acceleration_scaling_factor = 0.3
    r.goal_constraints = [Constraints(joint_constraints=[JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005, tolerance_below=0.005, weight=1.0) for j, v in zip(ARM_JOINTS, goal)])]
    res = _wait(cli.call_async(rq), 30.0).motion_plan_response
    if res.error_code.val != 1: print('plan failed', res.error_code.val); continue
    pts = [np.array(pt.positions) for pt in res.trajectory.joint_trajectory.points]
    names = list(res.trajectory.joint_trajectory.joint_names); idx = [names.index(j) for j in ARM_JOINTS]
    pts = [q[idx] for q in pts]
    apply_shelf(0.03)
    bad = []; maxstep = 0
    for a, b in zip(pts, pts[1:]):
        maxstep = max(maxstep, np.degrees(np.max(np.abs(b - a))))
        for t in np.linspace(0, 1, 11)[1:]:
            c = contacts(a + (b - a) * t)
            if c: bad.append((np.round(np.degrees(a + (b - a) * t)).astype(int).tolist(), c))
    print(f'{sname} -> D11 plan {trial}: {len(pts)} points, max step between points {maxstep:.1f} deg, colliding sub-steps {len(bad)}', bad[:3], flush=True)
    if bad: print('   first colliding at', bad[0])

apply_shelf(0.01); print('scene restored')
