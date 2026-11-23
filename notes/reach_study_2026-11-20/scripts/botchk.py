import math, numpy as np, rclpy
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, TCP
rclpy.init(); node = rclpy.create_node('botchk'); mv = MoveItClient(node, base_frame='base')
SEEDS = [(j2, j3, 0.0) for j2, j3 in SEEDS_J23] + [(-0.8,-1.2,0.4),(0.3,-1.8,0.0),(-1.2,-0.6,0.3),(0.6,-2.0,-0.2),(-1.2,0.4,-0.7)]
def ok(x, y, z, ty, coll):
    j1 = math.atan2(y, x)
    for k in (0, 1):
        T = make_T(rot_z(ty + math.pi/2 + k*math.pi) @ DOWN, np.array([x, y, z]))
        for a2, a3, a4 in SEEDS:
            q = mv.compute_ik(T, TCP, ARM_JOINTS, np.array([j1, a2, a3, a4, 0, 0]), timeout=0.05, avoid_collisions=coll)
            if q is not None and abs(math.remainder(q[0]-j1, 2*math.pi)) <= math.radians(60): return True
    return False
hx, hy, hz, ty = 0.2155, -0.0449, -0.208, math.radians(180)
for d in (0.0, 0.02, 0.04, 0.06, 0.08):
    x, y = hx, hy - d
    res = []
    for h in (0.02, 0.04, 0.06, 0.08):
        c = ok(x, y, hz + h, ty, True) if d == 0 else None
        k = ok(x, y, hz + h, ty, False)
        res.append(f'h{int(h*100)}:' + ('#' if c else 'Y' if k else '-'))
    print(f'robot +y {d*100:.0f} cm  handle ({x:+.3f},{y:+.3f},{hz:+.3f}) r {math.hypot(x,y)*100:.1f} cm  ' + ' '.join(res), flush=True)
