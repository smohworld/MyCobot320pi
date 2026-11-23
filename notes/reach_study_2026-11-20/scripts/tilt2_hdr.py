import math, heapq, sys, numpy as np
from scipy.optimize import least_squares
from fastik import arm, LO, HI
TILT = math.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
def ik_tilt(p, yaw, q0, tilt=None):
    tl = TILT if tilt is None else tilt
    """TCP at p, z axis within TILT of straight down, x axis heading yaw (in the horizontal), near q0."""
    def f(q):
        T = arm.fk(q); z = T[:3, 2]; x = T[:3, 0]
        ang = math.acos(max(-1.0, min(1.0, -z[2])))
        hx = math.atan2(x[1], x[0])
        return np.r_[(T[:3, 3] - p) * 10, max(0.0, ang - tl) * 3, math.remainder(hx - yaw, 2 * math.pi) * 0.5, (q - q0) * 0.02]
    r = least_squares(f, q0, bounds=(LO, HI))
    return r.x if np.linalg.norm(r.fun[:5]) < 4e-3 else None
D11 = np.array([0.003, 0.313, 0.029]); import yaml, os
_tb = yaml.safe_load(open(os.path.expanduser('~/.ros/layout_tray_bot.yaml')))
B = np.array(_tb['handle_xyz'], float); print('TurtleBot handle', B, 'yaw', _tb['yaw_lr_deg'], flush=True)
