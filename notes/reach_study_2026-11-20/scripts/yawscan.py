import math, numpy as np
from scipy.optimize import least_squares
from fastik import arm, LO, HI, DOWN
from mycobot_calibration.geometry import rot_z
from diag import check
rng = np.random.default_rng(1)
def sols(p, R, n=30):
    out = []
    def f(q):
        T = arm.fk(q); return np.r_[(T[:3, 3] - p) * 10, (T[:3, :3].T @ R - np.eye(3)).ravel() * 0.3]
    for _ in range(n):
        r = least_squares(f, rng.uniform(LO, HI), bounds=(LO, HI))
        if np.linalg.norm(r.fun) < 3e-3 and not any(np.allclose(r.x, o, atol=0.05) for o in out): out.append(r.x)
    return out
p = np.array([0.003, 0.313, 0.029 + 0.02]); j1 = math.degrees(math.atan2(p[1], p[0]))
for yd in range(0, 180, 15):
    R = rot_z(math.radians(yd)) @ DOWN
    res = []
    for q in sols(p, R):
        v, c = check(q); c = [x for x in c if x != ('base', 'desk')]
        res.append(f"J1 {math.degrees(q[0]):+.0f}{'' if not c else ' X' + ','.join(sorted({b for a in c for b in a if b.startswith(('shelf','desk','bot','turtle','robot'))}))}")
    print(f'tool yaw {yd:3d}: ' + ' | '.join(res), flush=True)
