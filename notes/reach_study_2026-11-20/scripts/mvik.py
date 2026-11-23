import math, numpy as np, rclpy
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import DOWN, TCP
from fastik import solve
rclpy.init(); node = rclpy.create_node('mvik'); mv = MoveItClient(node, base_frame='base')
for name, p, yaw in (('bot +4', (0.2155, -0.0449, -0.208 + 0.04), math.pi), ('D11 +2', (0.003, 0.313, 0.029 + 0.02), math.pi)):
    q0 = solve(np.array(p), yaw)
    T = make_T(rot_z(yaw + math.pi/2) @ DOWN, np.array(p))
    for coll in (True, False):
        q = mv.compute_ik(T, TCP, ARM_JOINTS, q0, timeout=1.0, avoid_collisions=coll)
        print(name, 'collision' if coll else 'no-collision', None if q is None else np.round(np.degrees(q)).astype(int), flush=True)
