import math, numpy as np
from swapik_lib import *
from scipy.optimize import least_squares
from fastik import arm, LO, HI, DOWN
from mycobot_calibration.geometry import rot_z
from route import contacts
rng = np.random.default_rng(0)
try:
    L.v.update(bot=[0.0, 0.22, 90.0], with_shelf=False); apply_all(0.01)
    for yawd in (0, 90, 180, 270):
        p = np.array([0.0, 0.22, -0.208 - 0.02]); R = rot_z(math.radians(yawd)) @ DOWN
        f = lambda q: np.r_[(arm.fk(q)[:3, 3] - p) * 10, (arm.fk(q)[:3, :3].T @ R - np.eye(3)).ravel() * 0.3]
        seen = []
        for _ in range(25):
            r = least_squares(f, rng.uniform(LO, HI), bounds=(LO, HI))
            if np.linalg.norm(r.fun) < 3e-3 and not any(np.allclose(r.x, s_, atol=0.1) for s_ in seen): seen.append(r.x)
        print(f'tool yaw {yawd}: {len(seen)} kinematic solutions;', [(np.round(np.degrees(q[:4])).astype(int).tolist(), contacts(q)) for q in seen[:4]], flush=True)
finally:
    L.v.update(orig); apply_all(0.01); print('scene restored')
