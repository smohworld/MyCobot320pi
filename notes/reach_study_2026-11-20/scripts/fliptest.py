import math, threading, time, numpy as np, rclpy
from rclpy.executors import MultiThreadedExecutor
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, TCP
rclpy.init(); node = rclpy.create_node('fliptest'); mv = MoveItClient(node, base_frame='base')
ex = MultiThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
SEEDS = [(j2, j3, 0.0) for j2, j3 in SEEDS_J23] + [(-0.8,-1.2,0.4),(0.3,-1.8,0.0),(-1.2,-0.6,0.3),(0.6,-2.0,-0.2),(-1.2,0.4,-0.7)]
def run(xy, hz, ty, label):
    t0 = time.time()
    for flip in (False, True):
        j1 = math.remainder(math.atan2(xy[1], xy[0]) + (math.pi if flip else 0), 2*math.pi); m = -1 if flip else 1
        for h in (0.04, 0.02):
            n = 0
            for k in (0, 1):
                T = make_T(rot_z(ty + math.pi/2 + k*math.pi) @ DOWN, np.array([xy[0], xy[1], hz + h]))
                for a2, a3, a4 in SEEDS:
                    q = mv.compute_ik(T, TCP, ARM_JOINTS, np.array([j1, m*a2, m*a3, m*a4, 0, 0]), timeout=0.05)
                    if q is not None and abs(math.remainder(q[0]-j1, 2*math.pi)) <= math.radians(60):
                        n += 1; qq = q
            print(f'{label} flip={flip} h={h}: {n} solutions' + (f' e.g. {np.round(np.degrees(qq)).astype(int)}' if n else ''), flush=True)
    print(f'  {time.time()-t0:.1f} s')
run((0.003, 0.313), -0.043 + 0.072, math.radians(180), 'D11')
run((0.2155, -0.0449), -0.280 + 0.072, math.radians(180), 'bot')
