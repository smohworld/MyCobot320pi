import math, sys, numpy as np, rclpy
from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import MoveItClient
from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, TCP
rclpy.init(); node = rclpy.create_node('r_sweep'); mv = MoveItClient(node, base_frame='base')
SEEDS = [(j2, j3, 0.0) for j2, j3 in SEEDS_J23] + [(-0.8,-1.2,0.4),(0.3,-1.8,0.0),(-1.2,-0.6,0.3),(0.6,-2.0,-0.2),(-1.2,0.4,-0.7)]
d11z = -0.002; handle = d11z + 0.072; tray_yaw = math.radians(91.03 + 90)
bearings = [float(b) for b in sys.argv[1:]] or [math.degrees(math.atan2(0.277, -0.116))]
def ok(x, y, z, coll):
    j1 = math.atan2(y, x)
    for k in (0, 1):
        T = make_T(rot_z(tray_yaw + math.pi/2 + k*math.pi) @ DOWN, np.array([x, y, z]))
        for a2, a3, a4 in SEEDS:
            q = mv.compute_ik(T, TCP, ARM_JOINTS, np.array([j1, a2, a3, a4, 0, 0]), timeout=0.05, avoid_collisions=coll)
            if q is not None and abs(math.remainder(q[0]-j1, 2*math.pi)) <= math.radians(60):
                return True
    return False
for b in bearings:
    print(f'bearing {b:.0f} deg   (h = finger tips over handle top; # = IK ok with collisions, o = kinematic only, . = no)')
    hs = [0.02, 0.04, 0.06, 0.08]
    print('  r cm  ' + '  '.join(f'h{int(h*100)}' for h in hs))
    for r in np.arange(0.12, 0.345, 0.01):
        x, y = r*math.cos(math.radians(b)), r*math.sin(math.radians(b))
        row = []
        for h in hs:
            row.append('#' if ok(x, y, handle+h, True) else ('o' if ok(x, y, handle+h, False) else '.'))
        print(f'  {r*100:4.0f}   ' + '   '.join(row), flush=True)
