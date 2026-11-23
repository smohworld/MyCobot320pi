import math, heapq, sys, numpy as np
from held import contacts_held, ik_local, ik_any, seg, chain, D11, B
Y0 = math.radians(270); b0 = math.atan2(D11[1], D11[0])
DESK_EDGE_Y = 0.0938; DESK_END_X = 0.0656; DESK_TIP = -0.007 + 0.072 - 0.02   # finger tips when the tray stands on the desk
ZC = 0.125
q0 = ik_any(D11 - [0, 0, 0.02], Y0, True)
p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069])
qs, w = chain(q0, seg(D11 - [0, 0, 0.02], D11 + [0, 0, 0.04]) + seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2) + seg(p2, [p2[0], p2[1], ZC]), Y0, skip=range(3))
print('D11 -> pulled out, up:', w, flush=True)
def on_desk(x, y, yaw):                         # tray footprint (15 along TCP x, 10 along TCP y) inside the desk top, 1 cm in
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(x + c * u - s * v, y + s * u + c * v) for u in (-0.075, 0.075) for v in (-0.05, 0.05)]
    return all(px <= DESK_END_X - 0.01 and py <= DESK_EDGE_Y - 0.01 for px, py in pts)
for mode in ('turn', '0', '-90', '90'):
    yawof = (lambda x, y: Y0 + (math.atan2(y, x) - b0)) if mode == 'turn' else (lambda x, y, o=math.radians(float(mode)): Y0 + o)
    q = qs
    if mode not in ('turn', '0'):                 # turn the tray in place first
        for t in np.linspace(0, 1, 30)[1:]:
            q = ik_local(np.array([p2[0], p2[1], ZC]), Y0 + math.radians(float(mode)) * t, q)
            if q is None: break
        if q is None or contacts_held(q): print(mode, 'turn in place failed'); continue
    S = (p2[0], p2[1])
    xs = np.round(np.arange(-0.30, 0.10, 0.02), 3); ys = np.round(np.arange(-0.30, 0.22, 0.02), 3)
    nodes = [(x, y) for x in xs for y in ys if math.hypot(x, y) >= 0.12]
    best = {S: 0.0}; qof = {S: q}; pq = [(0.0, S)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, 1e9) + 1e-9: continue
        for v in nodes:
            if v == u or math.dist(u, v) > 0.03 or d + math.dist(u, v) >= best.get(v, 1e9): continue
            qq = qof[u]; ok = True
            for p in seg([u[0], u[1], ZC], [v[0], v[1], ZC], step=0.005):
                qn = ik_local(p, yawof(p[0], p[1]), qq)
                if qn is None or np.degrees(np.max(np.abs(qn - qq))) > 12: ok = False; break
                qq = qn
            if not ok or contacts_held(qq): continue
            best[v] = d + math.dist(u, v); qof[v] = qq; heapq.heappush(pq, (best[v], v))
    spots = [v for v in best if on_desk(v[0], v[1], yawof(*v))]
    print(f'tray yaw mode {mode}: {len(best)} points reachable at z {ZC}, {len(spots)} with the tray on the desk', flush=True)
    found = 0
    for v in sorted(spots, key=lambda v: best[v]):
        yaw = yawof(*v)
        qd, w = chain(qof[v], seg([v[0], v[1], ZC], [v[0], v[1], DESK_TIP]), yaw, skip=())
        if qd is None:   # last cm: the tray touches the desk -> allow
            qd, w = chain(qof[v], seg([v[0], v[1], ZC], [v[0], v[1], DESK_TIP + 0.01]), yaw)
        if qd is None: continue
        qn = ik_any(np.array([v[0], v[1], DESK_TIP]), yaw, False)       # normal arm regrasp
        print(f'   spot ({v[0]:+.2f},{v[1]:+.2f}) tray yaw {math.degrees(yaw) % 360:.0f}: reversed put down OK; normal regrasp ' + ('OK' if qn is not None else 'no'), flush=True)
        found += 1
        if found >= 4: break
