import math, numpy as np
import chainlib
from chainlib import chain, seg
from scanlib import L, orig, ap, _wait
from route import ik_any
from mycobot_tasks import place_scene as ps
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene
def apply_all(m=0.01):
    a = L.args(); a.shelf_margin = m; o, _ = ps.build(a)
    req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True)); req.scene.world.collision_objects = ps.collision_objects(o, 'base')
    r = _wait(ap.call_async(req), 20.0); assert r and r.success
    return o
def xy_overlap(o, a, b):
    flat = lambda b: ([b[0][0], b[0][1], 0.0], [b[1][0], b[1][1], 1.0], b[2])
    return any(ps._overlap(flat(p), flat(q)) for p in o[a] for q in o[b])
# TurtleBot on the +y side, its box clear of the desk top seen from above
for by in np.arange(0.20, 0.32, 0.01):
    L.v['bot'] = [0.0, float(by), 90.0]; o, _ = ps.build(L.args())
    if not xy_overlap(o, 'bot_box', 'desk'): break
print('TurtleBot', L.v['bot'])
_, bi = ps.build(L.args())
cases = [(180.0, 0.097, 0.69, 0.36), (0.0, -0.097, 0.09, -0.36), (170.0, 0.097, 0.76, 0.28), (10.0, -0.097, 0.16, -0.35), (190.0, 0.097, 0.63, 0.44)]
try:
    for yaw, sx, ox, oy in cases:
        L.v.update(shelf=[ox, oy, yaw], shelf_x=sx, tray_yaw_off=90.0)
        o, _ = ps.build(L.args())
        if ps.box_hits(o, [k for k in o if k.startswith('shelf')], ['desk', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box']): print('overlap', yaw); continue
        x, y, z = L.fork_point(); D = np.array([x, y, z + 0.072])
        Y = math.radians(yaw + 90) + math.pi / 2; chainlib.YAW = Y
        bx, by_ = L.v['bot'][:2]; BOT = np.array([bx, by_, -0.208])
        res = []
        for m in (0.01, 0.02, 0.03):
            apply_all(m)
            g = ik_any(D - [0, 0, 0.02], Y, False); a_ = ik_any(D + [0, 0, 0.04], Y, False)
            gb = ik_any(BOT - [0, 0, 0.02], Y, False)
            w = '-'
            if g is not None and a_ is not None:
                P = D + [0, 0, 0.04]; Pp = P + [-0.10, 0, 0]; Pz = np.array([Pp[0], Pp[1], 0.125])   # pull toward the base (-x)
                _, w = chain(a_, seg(P, Pp) + seg(Pp, Pz) + seg(Pz, [BOT[0], BOT[1], 0.125], polar=True) + seg([BOT[0], BOT[1], 0.125], BOT + [0, 0, 0.04]) + seg(BOT + [0, 0, 0.04], BOT - [0, 0, 0.02]))
            res.append(f"{m*100:.0f}cm: D11 {'Y' if g is not None and a_ is not None else '-'} bot {'Y' if gb is not None else '-'} route {w.split(' [')[0]}")
        print(f'shelf {ox:+.2f},{oy:+.2f},{yaw:+.0f} x{sx:+.3f}  D11 ({x:+.3f},{y:+.3f}) r {math.hypot(x, y)*100:.1f}: ' + ' | '.join(res), flush=True)
finally:
    L.v.update(orig); apply_all(0.01); print('scene restored')
