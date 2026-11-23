import math, numpy as np
from scanlib import L, apply_shelf, valid, orig
from scipy.optimize import least_squares
from fastik import arm, LO, HI, DOWN
from mycobot_calibration.geometry import rot_z
rng = np.random.default_rng(5)
def sols(p, flip, n=12):
    R = rot_z(math.pi / 2) @ DOWN; out = []; b = math.atan2(p[1], p[0]) + (math.pi if flip else 0)
    f = lambda q: np.r_[(arm.fk(q)[:3, 3] - p) * 10, (arm.fk(q)[:3, :3].T @ R - np.eye(3)).ravel() * 0.3]
    for _ in range(n):
        q0 = rng.uniform(LO, HI); q0[0] = math.remainder(b + rng.normal(0, 0.3), 2 * math.pi)
        r = least_squares(f, np.clip(q0, LO, HI), bounds=(LO, HI))
        if np.linalg.norm(r.fun) < 3e-3 and abs(math.remainder(r.x[0] - b, 2 * math.pi)) < math.radians(60): out.append(r.x)
    return out
M = (0.01, 0.02, 0.03, 0.05)
hz = -0.043 + 0.072
try:
    cases = [('reversed', 0.0, True)] + [(f'normal, handle {d*100:.0f} cm toward robot', d, False) for d in (0.0, 0.02, 0.04, 0.06, 0.08)]
    for name, d, flip in cases:
        p_g = np.array([0.003, 0.313 - d, hz - 0.02]); p_a = p_g + [0, 0, 0.06]
        qg, qa = sols(p_g, flip), sols(p_a, flip)
        row = []
        for m in M:
            apply_shelf(m)
            row.append(f'{m*100:.0f}cm: grasp {"Y" if any(valid(q) for q in qg) else "-"} approach {"Y" if any(valid(q) for q in qa) else "-"}')
        print(f'{name:38s} (IK {len(qg)}/{len(qa)})  ' + ' | '.join(row), flush=True)
finally:
    L.v.update(orig); apply_shelf(0.01); print('scene restored')
