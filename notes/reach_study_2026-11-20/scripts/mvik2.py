import math, numpy as np, rclpy
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import TCP
from fastik import solve, arm
rclpy.init(); node = rclpy.create_node('mvik2'); mv = MoveItClient(node, base_frame='base')
import threading; from rclpy.executors import MultiThreadedExecutor; ex = MultiThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
for name, p in (('bot +4', (0.2155, -0.0449, -0.168)), ('D11 +2', (0.003, 0.313, 0.049)), ('easy', (0.2, 0.0, 0.1))):
    q0 = solve(np.array(p), math.pi)
    T = arm.fk(q0)       # exact pose of that solution
    print(name, 'fk', np.round(T[:3, 3], 3), 'J', ARM_JOINTS == list(ARM_JOINTS))
    for coll in (False, True):
        q = mv.compute_ik(T, TCP, ARM_JOINTS, q0, timeout=1.0, avoid_collisions=coll)
        print('  ', 'coll' if coll else 'nocoll', None if q is None else np.round(np.degrees(q)).astype(int), flush=True)
