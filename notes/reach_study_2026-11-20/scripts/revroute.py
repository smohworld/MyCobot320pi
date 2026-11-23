import math, numpy as np
import chainlib
from chainlib import chain, seg
from route import ik_any, contacts
D11 = np.array([0.003, 0.313, 0.029]); B = np.array([0.2155, -0.0449, -0.208])
for yd in (270, 90):
    Y = math.radians(yd); chainlib.YAW = Y
    q0 = ik_any(D11 + [0, 0, 0.04], Y, True)
    if q0 is None: print(yd, 'no start'); continue
    p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069]); p3 = np.array([0.023, 0.16, 0.125])
    qa, w = chain(q0, seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2) + seg(p2, p3))
    print(f'tool yaw {yd}: D11 -> pulled out + up: {w}', flush=True)
    if qa is None: continue
    for mid in ((0.10, 0.15), (0.15, 0.12), (0.18, 0.08), (0.20, 0.05)):
        M = np.array([mid[0], mid[1], 0.125])
        q, w = chain(qa, seg(p3, M) + seg(M, [B[0], B[1], 0.125], polar=True) + seg([B[0], B[1], 0.125], B + [0, 0, 0.04]))
        print(f'   via ({mid[0]:+.2f},{mid[1]:+.2f}) -> bot +4 cm: {w}' + ('' if q is None else f'  end q {np.round(np.degrees(q)).astype(int)}'), flush=True)
B2 = B
for h in (-0.02, -0.01, 0.0, 0.02):
    q = None
    from route import ik_local
    import fastik
    q = fastik.solve(B + [0, 0, h], math.radians(270))
    print(f'bot tips {h*100:+.0f} cm: kinematic', 'Y' if q is not None else '-', '' if q is None else contacts(q))
