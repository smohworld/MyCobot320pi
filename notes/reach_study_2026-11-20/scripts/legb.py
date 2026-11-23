import math, heapq, sys, numpy as np
sys.argv = [sys.argv[0], '10']
src = open('tilt2.py').read(); src = src[:src.index('q0 = ik_any(D11')]
exec(src)
Yc = Y0 + math.radians(90)                              # tray yaw on the desk (as put down)
spots = [(-0.16, -0.14), (-0.14, -0.16), (-0.12, -0.18), (-0.18, -0.12), (-0.20, -0.08), (-0.10, -0.20)]
G = (B[0], B[1])
xs = np.round(np.arange(-0.30, 0.30, 0.02), 3); ys = np.round(np.arange(-0.30, 0.22, 0.02), 3)
nodes = [(x, y) for x in xs for y in ys if math.hypot(x, y) >= 0.12] + [G]
for sx, sy in spots:
    qg = ik_any(np.array([sx, sy, DESK_TIP]), Yc, False)
    if qg is None: print(f'spot ({sx:+.2f},{sy:+.2f}): no normal grasp'); continue
    q, w = chain(qg, seg([sx, sy, DESK_TIP], [sx, sy, ZC]), Yc, skip=range(4))
    if q is None: print(f'spot ({sx:+.2f},{sy:+.2f}): lift {w}'); continue
    for yb in (90, 270, 0, 180):                          # tray yaw for the box (box heading 270 -> 90/270)
        Yb = math.radians(yb); qq = q; d = math.remainder(Yb - Yc, 2 * math.pi)
        for t in np.linspace(0, 1, 30)[1:]:
            qq = ik_tilt(np.array([sx, sy, ZC]), Yc + d * t, qq) if qq is not None else None
        if qq is None or contacts_held(qq): print(f'spot ({sx:+.2f},{sy:+.2f}) tray {yb}: turn failed'); continue
        S = (sx, sy); best = {S: 0.0}; qof = {S: qq}; pq = [(0.0, S)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > best.get(u, 1e9) + 1e-9: continue
            if u == G: break
            for v in nodes:
                if v == u or math.dist(u, v) > 0.03 or dd + math.dist(u, v) >= best.get(v, 1e9): continue
                qa = qof[u]; ok = True
                for p in seg([u[0], u[1], ZC], [v[0], v[1], ZC], step=0.005):
                    qn = ik_tilt(p, Yb, qa)
                    if qn is None or np.degrees(np.max(np.abs(qn - qa))) > 12: ok = False; break
                    qa = qn
                if not ok or contacts_held(qa): continue
                best[v] = dd + math.dist(u, v); qof[v] = qa; heapq.heappush(pq, (best[v], v))
        if G not in best: print(f'spot ({sx:+.2f},{sy:+.2f}) tray {yb}: no path to the bot ({len(best)} points)', flush=True); continue
        qv = qof[G]; why = None
        for t in np.linspace(0, 1, 11)[1:]:
            qn = ik_tilt(np.array([G[0], G[1], ZC]), Yb, qv, tilt=TILT * (1 - t) + 1e-3)
            if qn is None or np.degrees(np.max(np.abs(qn - qv))) > 12: why = 'straighten failed'; break
            qv = qn
        if why is None:
            qd, why = chain(qv, seg([G[0], G[1], ZC], B + [0, 0, 0.0]), Yb)
        print(f'spot ({sx:+.2f},{sy:+.2f}) tray {yb}: PATH to the bot found, down into the box: {why}', flush=True)
