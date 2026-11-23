# Is there a vertical grasp at the bot handle (+hs) at all? kinematics only, then collision with the held tray.
import math, numpy as np
from route import ik_local
from fastik import arm, LO, HI
from held import contacts_held
from tilt2_hdr import B
rng = np.random.default_rng(1)
for yd in (90, 270):
    Y = math.radians(yd)
    for hs in (0.08, 0.05, 0.02, 0.0):
        p = B + [0, 0, hs]; sols = []
        for _ in range(300):
            q0 = rng.uniform(LO, HI)
            q = ik_local(p, Y, q0)
            if q is not None and not any(np.max(np.abs(q - s)) < 0.05 for s in sols): sols.append(q)
        out = []
        for q in sols:
            c = contacts_held(q)
            out.append(f"J {np.round(np.degrees(q)).astype(int).tolist()} {'FREE' if not c else c[:3]}")
        print(f'tray yaw {yd} +{hs*100:.0f} cm: {len(sols)} IK branches', flush=True)
        for s in out: print('    ', s, flush=True)
