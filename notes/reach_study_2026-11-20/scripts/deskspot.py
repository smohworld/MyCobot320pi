import math, numpy as np
from route import ik_any
hz = -0.007 + 0.072                     # tray on the desk top: handle top
res = []
for x in np.arange(-0.25, 0.051, 0.05):
    for y in np.arange(-0.35, 0.081, 0.05):
        if math.hypot(x, y) < 0.12: continue
        ok = {}
        for flip in (True, False):
            ok[flip] = all(ik_any(np.array([x, y, hz + h]), math.pi + math.pi / 2, flip) is not None for h in (-0.02, 0.04))
        if ok[True] and ok[False]: res.append((x, y)); print(f'both arms OK at desk ({x:+.2f},{y:+.2f}) r {math.hypot(x, y)*100:.0f} cm', flush=True)
print('done', len(res))
