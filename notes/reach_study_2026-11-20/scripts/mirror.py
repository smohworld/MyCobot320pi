import math, numpy as np
from scanlib import L, apply_shelf, valid, orig
from mycobot_tasks import place_scene as ps
from route import ik_any, contacts
from relay import chain, seg, YAW, BOT
cands = []
for oy in np.arange(0.69, 0.75, 0.01):
    for ox in np.arange(-0.42, -0.20, 0.02):
        L.v.update(shelf=[float(ox), float(oy), -90.0], shelf_x=0.097)
        a = L.args(); o, _ = ps.build(a)
        if ps.box_hits(o, [k for k in o if k.startswith('shelf')], ['desk', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box']): continue
        x, y, z = L.fork_point(); cands.append((math.hypot(x, y), float(ox), float(oy), x, y, z))
cands.sort(); print('clear placements (best 3):', [tuple(round(v, 3) for v in c) for c in cands[:3]])
try:
    for r, ox, oy, x, y, z in cands[:3]:
        L.v.update(shelf=[ox, oy, -90.0], shelf_x=0.097)
        D = np.array([x, y, z + 0.072])
        for m in (0.01, 0.03):
            apply_shelf(m)
            g = ik_any(D - [0, 0, 0.02], YAW, False); ap = ik_any(D + [0, 0, 0.04], YAW, False)
            print(f'shelf {ox:.2f},{oy:.2f},-90  D11 ({x:+.3f},{y:+.3f}) r {r*100:.1f}  margin {m*100:.0f} cm: normal grasp {"Y" if g is not None else "-"} approach {"Y" if ap is not None else "-"}', flush=True)
        apply_shelf(0.01)
        if g is None or ap is None: continue
        # D11 -> TurtleBot with the normal arm: up, pull toward the base (-y), swing, down
        P = D + [0, 0, 0.04]; Pp = np.array([P[0], 0.17, P[2]]); Pz = np.array([P[0], 0.17, 0.125])
        q, w = chain(ap, seg(P, Pp) + seg(Pp, Pz) + seg(Pz, [BOT[0], BOT[1], 0.125], polar=True) + seg([BOT[0], BOT[1], 0.125], BOT + [0, 0, 0.04]) + seg(BOT + [0, 0, 0.04], BOT - [0, 0, 0.02]))
        print('   D11 -> TurtleBot (normal arm, no switch):', w, flush=True)
finally:
    L.v.update(orig); apply_shelf(0.01); print('scene restored')
