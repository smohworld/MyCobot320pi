import math, numpy as np
from route import ik_local, ik_any, valid, contacts
YAW = math.pi + math.pi / 2
D11 = np.array([0.003, 0.313, 0.029]); BOT = np.array([0.2155, -0.0449, -0.208])
DESK_H = -0.007 + 0.072
ZC = 0.125                                    # carry height of the finger tips
def seg(a, b, polar=False, step=0.005):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if polar:
        ra, rb = math.hypot(*a[:2]), math.hypot(*b[:2]); ba, bb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
        d = math.remainder(bb - ba, 2 * math.pi); L = abs(d) * max(ra, rb) + abs(rb - ra) + abs(b[2] - a[2])
        n = max(2, int(L / step))
        return [np.array([(ra + (rb - ra) * t) * math.cos(ba + d * t), (ra + (rb - ra) * t) * math.sin(ba + d * t), a[2] + (b[2] - a[2]) * t]) for t in np.linspace(0, 1, n)[1:]]
    n = max(2, int(np.linalg.norm(b - a) / step))
    return [a + (b - a) * t for t in np.linspace(0, 1, n)[1:]]
def chain(q, pts):
    for i, p in enumerate(pts):
        qn = ik_local(p, YAW, q)
        if qn is None: return None, f'no IK at {np.round(p, 3)}'
        if np.degrees(np.max(np.abs(qn - q))) > 12: return None, f'branch jump at {np.round(p, 3)}'
        if i % 2 == 0 or i == len(pts) - 1:
            if not valid(qn): return None, f'collision at {np.round(p, 3)} {contacts(qn)}'
        q = qn
    return q, 'ok'
