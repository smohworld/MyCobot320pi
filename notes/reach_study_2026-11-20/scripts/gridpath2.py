import math, heapq, numpy as np
from held import contacts_held, ik_local, ik_any, seg, chain, D11, B
Y0 = math.radians(270); b0 = math.atan2(D11[1], D11[0])
import sys
OFF = math.radians(float(sys.argv[1]))
yawof = lambda x, y: Y0 + OFF
q0 = ik_any(D11 - [0, 0, 0.02], Y0, True)
lift = seg(D11 - [0, 0, 0.02], D11 + [0, 0, 0.04]); p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069])
qs, w = chain(q0, lift + seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2), Y0, skip=range(3))
print('pull out:', w, flush=True)
for zc in (0.125,):
    start = (round(p2[0], 2), round(p2[1], 2))
    qz, w = chain(qs, seg(p2, [p2[0], p2[1], zc]), Y0)
    if qz is not None:
        q = qz
        for t in np.linspace(0, 1, 30)[1:]:
            q = ik_local(np.array([p2[0], p2[1], zc]), Y0 + OFF * t, q)
            if q is None: break
        qz = q if q is not None and not contacts_held(q) else None
        w = 'turn in place ' + ('ok' if qz is not None else 'failed')
        print(w, flush=True)
    if qz is None: print(f'z {zc}: up {w}'); continue
    # Dijkstra over a 2 cm grid; each edge checked by chained IK with the tray (yaw follows the bearing)
    S = (p2[0], p2[1]); goal = (B[0], B[1])
    xs = np.round(np.arange(-0.04, 0.30, 0.02), 3); ys = np.round(np.arange(-0.10, 0.22, 0.02), 3)
    nodes = [(x, y) for x in xs for y in ys if math.hypot(x, y) >= 0.12] + [goal]
    best = {S: 0.0}; qof = {S: qz}; prev = {}; pq = [(0.0, S)]; reached = None; tried = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, 1e9) + 1e-9: continue
        if math.dist(u, goal) < 1e-6: reached = u; break
        for v in nodes:
            if v == u or math.dist(u, v) > 0.03: continue
            if d + math.dist(u, v) >= best.get(v, 1e9): continue
            q = qof[u]; ok = True
            for p in seg([u[0], u[1], zc], [v[0], v[1], zc], step=0.005):
                qn = ik_local(p, yawof(p[0], p[1]), q)
                if qn is None or np.degrees(np.max(np.abs(qn - q))) > 12: ok = False; break
                q = qn
            tried += 1
            if not ok or contacts_held(q): continue
            best[v] = d + math.dist(u, v); qof[v] = q; prev[v] = u; heapq.heappush(pq, (best[v], v))
    if reached is None:
        print(f'z {zc}: no path ({tried} edges tried, {len(best)} grid points reachable)', flush=True); continue
    path = [reached]
    while path[-1] in prev: path.append(prev[path[-1]])
    path = path[::-1]
    print(f'z {zc}: PATH {len(path)} points:', [tuple(round(c, 2) for c in p) for p in path], flush=True)
    qd, w = chain(qof[reached], seg([B[0], B[1], zc], B + [0, 0, 0.04]), yawof(*B[:2]))
    print('   down to bot +4 cm:', w, flush=True)
    break
