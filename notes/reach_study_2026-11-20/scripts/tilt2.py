import math, heapq, sys, numpy as np
from scipy.optimize import least_squares
from fastik import arm, LO, HI
from held import contacts_held, ik_local, ik_any, seg, chain
TILT = math.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
def ik_tilt(p, yaw, q0, tilt=None):
    tl = TILT if tilt is None else tilt
    """TCP at p, z axis within TILT of straight down, x axis heading yaw (in the horizontal), near q0."""
    def f(q):
        T = arm.fk(q); z = T[:3, 2]; x = T[:3, 0]
        ang = math.acos(max(-1.0, min(1.0, -z[2])))
        hx = math.atan2(x[1], x[0])
        return np.r_[(T[:3, 3] - p) * 10, max(0.0, ang - tl) * 3, math.remainder(hx - yaw, 2 * math.pi) * 0.5, (q - q0) * 0.02]
    r = least_squares(f, q0, bounds=(LO, HI))
    return r.x if np.linalg.norm(r.fun[:5]) < 4e-3 else None
D11 = np.array([0.003, 0.313, 0.029]); import yaml, os
_tb = yaml.safe_load(open(os.path.expanduser('~/.ros/layout_tray_bot.yaml')))
B = np.array(_tb['handle_xyz'], float); print('TurtleBot handle', B, 'yaw', _tb['yaw_lr_deg'], flush=True)
DESK_EDGE_Y, DESK_END_X = 0.0938, 0.0656; DESK_TIP = -0.007 + 0.072 - 0.02; ZC = 0.125
Y0 = math.radians(270)
def on_desk(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return all(x + c * u - s * v <= DESK_END_X - 0.01 and y + s * u + c * v <= DESK_EDGE_Y - 0.01 for u in (-0.075, 0.075) for v in (-0.05, 0.05))
q0 = ik_any(D11 - [0, 0, 0.02], Y0, True)
p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069])
qs, w = chain(q0, seg(D11 - [0, 0, 0.02], D11 + [0, 0, 0.04]) + seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2) + seg(p2, [p2[0], p2[1], ZC]), Y0, skip=range(3))
print(f'tilt {math.degrees(TILT):.0f}: D11 -> pulled out, up: {w}', flush=True)
for off in (90,):
    Yc = Y0 + math.radians(off); q = qs
    for t in np.linspace(0, 1, 30)[1:]:
        q = ik_local(np.array([p2[0], p2[1], ZC]), Y0 + math.radians(off) * t, q) if q is not None else None
    if q is None or contacts_held(q): print(f'tray +{off}: turn in place failed'); continue
    S = (p2[0], p2[1]); G = (B[0], B[1])
    xs = np.round(np.arange(-0.30, 0.30, 0.02), 3); ys = np.round(np.arange(-0.30, 0.22, 0.02), 3)
    nodes = [(x, y) for x in xs for y in ys if math.hypot(x, y) >= 0.12] + [G]
    best = {S: 0.0}; qof = {S: q}; pq = [(0.0, S)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, 1e9) + 1e-9: continue
        for v in nodes:
            if v == u or math.dist(u, v) > 0.03 or d + math.dist(u, v) >= best.get(v, 1e9): continue
            qq = qof[u]; ok = True
            for p in seg([u[0], u[1], ZC], [v[0], v[1], ZC], step=0.005):
                qn = ik_tilt(p, Yc, qq)
                if qn is None or np.degrees(np.max(np.abs(qn - qq))) > 12: ok = False; break
                qq = qn
            if not ok or contacts_held(qq): continue
            best[v] = d + math.dist(u, v); qof[v] = qq; heapq.heappush(pq, (best[v], v))
    spots = [v for v in best if on_desk(v[0], v[1], Yc)]
    print(f'tray +{off}: {len(best)} points reachable, bot above reached: {G in best}, {len(spots)} desk spots', flush=True)
    if G in best:
        qd, w = chain(qof[G], seg([G[0], G[1], ZC], B + [0, 0, 0.04]), Yc)
        print(f'   DIRECT: down to bot +4 cm (vertical, tray yaw {math.degrees(Yc) % 360:.0f}): {w}', flush=True)
    n = 0
    for v in sorted(spots, key=lambda v: abs(math.hypot(*v) - 0.22)):
        qv = qof[v]; why = None
        for t in np.linspace(0, 1, 11)[1:]:                         # straighten up gradually
            qn = ik_tilt(np.array([v[0], v[1], ZC]), Yc, qv, tilt=TILT * (1 - t) + 1e-3)
            if qn is None or np.degrees(np.max(np.abs(qn - qv))) > 12: why = f'straightening failed at {t:.1f}'; break
            qv = qn
        if why is None:
            qn = ik_local(np.array([v[0], v[1], ZC]), Yc, qv)
            if qn is None or np.degrees(np.max(np.abs(qn - qv))) > 12: why = 'no exact vertical pose'
            else: qv = qn
        if why: pass
        else:
            qd, why = chain(qv, seg([v[0], v[1], ZC], [v[0], v[1], DESK_TIP + 0.01]), Yc)
            if qd is not None:
                qn = ik_any(np.array([v[0], v[1], DESK_TIP]), Yc, False)
                why = 'put down OK, normal regrasp ' + ('OK' if qn is not None else 'NO')
        print(f'   relay ({v[0]:+.2f},{v[1]:+.2f}) r {math.hypot(*v)*100:.0f}: {why}', flush=True); n += 1
        if n >= 16: break
