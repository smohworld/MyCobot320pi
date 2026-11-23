import math, heapq, sys, numpy as np
sys.argv = [sys.argv[0], '10']
src = open('tilt2.py').read(); src = src[:src.index('q0 = ik_any(D11')]
exec(src)
Yc = Y0 + math.radians(90)                              # tray yaw on the desk (as put down)
spots = [(-0.16, -0.14), (-0.14, -0.16), (-0.12, -0.18), (-0.18, -0.12), (-0.20, -0.08), (-0.10, -0.20)]
G = (B[0], B[1])
xs = np.round(np.arange(-0.30, 0.30, 0.02), 3); ys = np.round(np.arange(-0.30, 0.22, 0.02), 3)
nodes = [(x, y) for x in xs for y in ys if math.hypot(x, y) >= 0.12] + [G]
for sx, sy, gflip, yb in [(x, y, g, b) for (x, y) in spots for g in (0, 180) for b in (90, 270)]:
    Yg = Yc + math.radians(gflip); tag = f'spot ({sx:+.2f},{sy:+.2f}) grip {gflip} box {yb}'
    qg = ik_any(np.array([sx, sy, DESK_TIP]), Yg, False)
    if qg is None: print(f'{tag}: no normal grasp'); continue
    q, w = chain(qg, seg([sx, sy, DESK_TIP], [sx, sy, ZC]), Yg, skip=range(4))
    if q is None: print(f'{tag}: lift {w}'); continue
    if True:
        Yb = math.radians(yb); qq = q; d = math.remainder(Yb - Yg, 2 * math.pi)
        if abs(abs(d) - math.pi) < 1e-6: continue
        for t in np.linspace(0, 1, 30)[1:]:
            qq = ik_tilt(np.array([sx, sy, ZC]), Yg + d * t, qq) if qq is not None else None
        if qq is None or contacts_held(qq): print(f'{tag}: turn {math.degrees(d):+.0f} failed (J6 at grip {math.degrees(qg[5]):.0f})', flush=True); continue
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
        if G not in best: print(f'{tag}: no path to the bot ({len(best)} points)', flush=True); continue
        res = []
        for hs in (0.08, 0.05, 0.02):                      # go down tilted, straighten lower, then straight down
            qv, why = qof[G], None
            for p in seg([G[0], G[1], ZC], B + [0, 0, hs]):
                qn = ik_tilt(p, Yb, qv)
                if qn is None or np.degrees(np.max(np.abs(qn - qv))) > 12: why = f'tilted descent failed at {np.round(p, 3)}'; break
                if contacts_held(qn): why = f'collision on the tilted descent at {np.round(p, 3)} {contacts_held(qn)}'; break
                qv = qn
            if why is None:
                T = arm.fk(qv); tl = math.degrees(math.acos(-T[2, 2])); hx = math.degrees(math.atan2(T[1, 0], T[0, 0]))
                if False: print(f'   +{hs*100:.0f}: tilted q {np.round(np.degrees(qv)).astype(int).tolist()} tilt {tl:.1f} heading {hx:.0f} pos {np.round(T[:3,3],3)}', flush=True)
                qs = ik_local(B + [0, 0, hs], Yb, qv)
                if False: print(f'      exact vertical seeded from it: ' + ('none' if qs is None else f'{np.round(np.degrees(qs)).astype(int).tolist()} max dJ {np.degrees(np.max(np.abs(qs-qv))):.1f}'), flush=True)
                RR = []
                for t in np.linspace(0, 1, 11)[1:]:
                    qn = ik_tilt(B + [0, 0, hs], Yb, qv, tilt=TILT * (1 - t) + 1e-3)
                    if qn is None or np.degrees(np.max(np.abs(qn - qv))) > 12: why = f'straighten failed at +{hs*100:.0f} cm t {t:.1f} ' + ('no IK' if qn is None else f'jump {np.degrees(np.max(np.abs(qn - qv))):.1f}'); break
                    qv = qn
            if why is None:
                qd, why = chain(qv, seg(B + [0, 0, hs], B + [0, 0, 0.0]), Yb)
                why = f'straighten at +{hs*100:.0f} cm, down into the box: {why}'
            res.append(why)
            if 'box: ok' in why: break
        why = ' / '.join(res)
        print(f'{tag}: PATH to the bot found, down into the box: {why}', flush=True)
