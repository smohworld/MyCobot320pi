import math, itertools, numpy as np, sys
from mycobot_tasks import layout as lay, place_scene as ps
L = lay.Layout('~/.ros/layout.yaml')
objs, info = ps.build(L.args()); print('names', list(objs), 'margin', L.base.margin)
def corners(c, s, yaw):
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return [np.array(c[:2]) + R @ np.array([sx*s[0]/2, sy*s[1]/2]) for sx in (-1,1) for sy in (-1,1)], R
def hit(a, b):
    (ca, sa, ya), (cb, sb, yb) = a, b
    if abs(ca[2]-cb[2]) >= (sa[2]+sb[2])/2: return False
    pa, Ra = corners(ca, sa, ya); pb, Rb = corners(cb, sb, yb)
    for ax in list(Ra.T) + list(Rb.T):
        A = [p @ ax for p in pa]; B = [p @ ax for p in pb]
        if max(A) <= min(B) or max(B) <= min(A): return False
    return True
fixed = [n for n in objs if not n.startswith('shelf') and n != 'floor']
def collides(v):
    L.v.update(v); o, _ = ps.build(L.args())
    bad = set()
    for n in o:
        if not n.startswith('shelf'): continue
        for f in fixed:
            if any(hit(a, b) for a in o[n] for b in o[f]): bad.add((n, f))
    return bad
print('current:', collides({}))
best = []
for oy in np.arange(-0.10, 0.20, 0.005):
    for ox in np.arange(0.0, 0.70, 0.01):
        for fy in (0.0, 0.08, 0.16):
            v = dict(shelf=[float(ox), float(oy), 90.0], fork_y=fy)
            x, y, z = (L.v.update(v) or L.fork_point())
            r = math.hypot(x, y)
            if r > 0.36: continue
            if not collides(v): best.append((r, ox, oy, fy, x, y))
best.sort()
for b in best[:15]: print('r %.3f  shelf %.3f,%.3f,90  fork_y %.2f  D11 (%+.3f,%+.3f)' % b)
