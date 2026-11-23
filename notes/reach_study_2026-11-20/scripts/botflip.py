import math, numpy as np
from scanlib import n, apply_shelf
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, TCP
mv = MoveItClient(n, base_frame='base')
SEEDS = [(j2, j3, 0.0) for j2, j3 in SEEDS_J23] + [(-0.8,-1.2,0.4),(0.3,-1.8,0.0),(-1.2,-0.6,0.3),(0.6,-2.0,-0.2),(-1.2,0.4,-0.7)]
hx, hy, hz = 0.2155, -0.0449, -0.208
for flip in (False, True):
    jc = math.remainder(math.atan2(hy, hx) + (math.pi if flip else 0), 2 * math.pi); m = -1 if flip else 1
    for name, h in (('grasp', -0.02), ('approach', 0.04), ('above', 0.08)):
        found = []
        for k in (0, 1):
            T = make_T(rot_z(math.pi + math.pi / 2 + k * math.pi) @ DOWN, np.array([hx, hy, hz + h]))
            for a2, a3, a4 in SEEDS:
                q = mv.compute_ik(T, TCP, ARM_JOINTS, np.array([jc, m*a2, m*a3, m*a4, 0, 0]), timeout=0.1)
                if q is not None and abs(math.remainder(q[0] - jc, 2 * math.pi)) < math.radians(60):
                    found.append(np.round(np.degrees(q)).astype(int)); break
            if found: break
        print(f"{'reversed' if flip else 'normal  '} bot {name:8s}: {'OK ' + str(found[0]) if found else 'no IK (collision-checked)'}", flush=True)
