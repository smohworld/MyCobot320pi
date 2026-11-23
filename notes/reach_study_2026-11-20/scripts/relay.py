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
# leg A: reversed arm, D11 -> relay
qA0 = ik_any(D11 + [0, 0, 0.04], YAW, True)
p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069])
qA, why = chain(qA0, seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2))
print('D11 -> pulled out (reversed):', why, flush=True)
spots = [(0.05, -0.15), (0.05, -0.20), (0.05, -0.25), (0.0, -0.25), (0.05, -0.30), (-0.05, -0.25)]
up = np.array([p2[0], p2[1], ZC])
qU, wU = chain(qA, seg(p2, up))
print('lift at pulled point:', wU)
for rs in (0.20, 0.24, 0.28):
    b2 = math.atan2(p2[1], p2[0]); W = np.array([rs * math.cos(b2), rs * math.sin(b2), ZC])
    qW, wW = chain(qU, seg(up, W))
    if qW is None: print(f'r_swing {rs}: out to it {wW}'); continue
    for sx, sy in spots:
        S = np.array([sx, sy, DESK_H])
        q1, w1 = chain(qW, seg(W, [sx, sy, ZC], polar=True) + seg([sx, sy, ZC], S + [0, 0, 0.04]) + seg(S + [0, 0, 0.04], S - [0, 0, 0.02]))
        if q1 is None: print(f'r_swing {rs} spot ({sx:+.2f},{sy:+.2f}): leg A {w1}', flush=True); continue
        qB0 = ik_any(S - [0, 0, 0.02], YAW, False)
        q2, w2 = (None, 'no normal grasp') if qB0 is None else chain(qB0, seg(S - [0, 0, 0.02], [sx, sy, ZC]) + seg([sx, sy, ZC], [BOT[0], BOT[1], ZC], polar=True) + seg([BOT[0], BOT[1], ZC], BOT + [0, 0, 0.04]) + seg(BOT + [0, 0, 0.04], BOT - [0, 0, 0.02]))
        print(f'r_swing {rs} spot ({sx:+.2f},{sy:+.2f}): leg A OK, leg B {w2}', flush=True)
