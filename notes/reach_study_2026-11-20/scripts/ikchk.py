import math, numpy as np, rclpy
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, TCP
rclpy.init(); node = rclpy.create_node('ikchk'); mv = MoveItClient(node, base_frame='base')
SEEDS = [(j2, j3, 0.0) for j2, j3 in SEEDS_J23] + [(-0.8,-1.2,0.4),(0.3,-1.8,0.0),(-1.2,-0.6,0.3),(0.6,-2.0,-0.2),(-1.2,0.4,-0.7)]
floor = -0.306
def ok(x, y, z, ty):
    j1 = math.atan2(y, x)
    for k in (0, 1):
        T = make_T(rot_z(ty + math.pi/2 + k*math.pi) @ DOWN, np.array([x, y, z]))
        for a2, a3, a4 in SEEDS:
            q = mv.compute_ik(T, TCP, ARM_JOINTS, np.array([j1, a2, a3, a4, 0, 0]), timeout=0.05, avoid_collisions=False)
            if q is not None and abs(math.remainder(q[0]-j1, 2*math.pi)) <= math.radians(60): return True
    return False
ty = math.radians(180)   # shelf yaw 90 + tray_yaw_off 90
pts = [(0.023, y) for y in (0.253, 0.27, 0.29, 0.31, 0.33)] + [(0.10, 0.253), (-0.05, 0.253)]
for ft in (0.263, 0.287, 0.304):
    hz = floor + ft + 0.072
    for x, y in pts:
        print(f'fork_top {ft*1000:.0f}  D11 ({x:+.3f},{y:+.3f}) r {math.hypot(x,y)*100:.1f}  ' +
              ' '.join(f'h{int(h*100)}:' + ('Y' if ok(x, y, hz+h, ty) else '-') for h in (0.02, 0.04, 0.06, 0.08)), flush=True)
