import math, numpy as np, rclpy
from diag import check
from fastik import arm
from mycobot_calibration.kin_calib import Arm, get_urdf
u = get_urdf(None)
print('q=0:', check(np.zeros(6)))
q = np.radians([106, -39, -49, -2, 90, 106])
for l in ('link2', 'link3', 'link4', 'link5', 'link6', 'gripper_tcp'):
    print(l, np.round(Arm(u, tip=l).fk(q)[:3, 3], 3))
