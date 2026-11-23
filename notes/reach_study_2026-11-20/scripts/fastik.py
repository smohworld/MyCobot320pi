import math, sys, numpy as np
from scipy.optimize import least_squares
from mycobot_calibration.kin_calib import Arm, get_urdf
from mycobot_calibration.geometry import make_T, rot_z
DOWN = np.diag([1.0, -1.0, -1.0])
import os
_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robot.urdf')   # xacro output (no calibration_file)
arm = Arm(get_urdf(_URDF if os.path.exists(_URDF) else None), tip='gripper_tcp')
LO = np.array([-2.93, -2.35, -2.53, -2.53, -2.93, -3.14]); HI = -LO
rng = np.random.default_rng(0)
def solve(p, yaw):
    """-> (q, link z list) for tool-down at p, fingers yaw+90 (either way), or None"""
    best = None
    for k in (0, 1):
        R = rot_z(yaw + math.pi/2 + k*math.pi) @ DOWN
        def f(q):
            T = arm.fk(q); e = T[:3, 3] - p
            dR = T[:3, :3].T @ R
            return np.r_[e * 10, (dR - np.eye(3)).ravel() * 0.3]
        for _ in range(25):
            q0 = rng.uniform(LO, HI); q0[0] = math.atan2(p[1], p[0]) + rng.normal(0, 0.3)
            r = least_squares(f, np.clip(q0, LO, HI), bounds=(LO, HI))
            if np.linalg.norm(r.fun[:3]) / 10 < 1e-3 and np.linalg.norm(r.fun[3:]) / 0.3 < 1e-2:
                return r.x
    return None
if __name__ == '__main__':
    print(arm.fk(np.zeros(6))[:3, 3])
