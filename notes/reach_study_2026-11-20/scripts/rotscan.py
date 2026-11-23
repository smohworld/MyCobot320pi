import math, sys, numpy as np
from scanlib import L, apply_shelf, orig
from mycobot_tasks import place_scene as ps
from route import ik_any
import chainlib
from chainlib import chain, seg, BOT
base_yaw, sx = float(sys.argv[1]), float(sys.argv[2])
def best_place(yaw):
    c = []
    for oy in np.arange(-0.2, 0.80, 0.01):
        for ox in np.arange(-0.5, 0.6, 0.02):
            L.v.update(shelf=[float(ox), float(oy), yaw], shelf_x=sx)
            x, y, z = L.fork_point()
            if math.hypot(x, y) > 0.315 or y < 0.15: continue
            o, _ = ps.build(L.args())
            if ps.box_hits(o, [k for k in o if k.startswith('shelf')], ['desk', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box']): continue
            c.append((math.hypot(x, y), float(ox), float(oy)))
    return sorted(c)[:2]
try:
    for dy in (-10, -5, 0, 5, 10, 15):
        yaw = base_yaw + dy
        for r, ox, oy in best_place(yaw):
            L.v.update(shelf=[ox, oy, yaw], shelf_x=sx); x, y, z = L.fork_point()
            ty = math.radians(yaw + 90)               # tray across the fork as before (tray_yaw_off 90)
            Y = ty + math.pi / 2; chainlib.YAW = Y
            D = np.array([x, y, z + 0.072]); out = []
            for m in (0.01, 0.02, 0.03):
                apply_shelf(m)
                g = ik_any(D - [0, 0, 0.02], Y, False); ap = ik_any(D + [0, 0, 0.04], Y, False)
                ok = g is not None and ap is not None
                route = ''
                if ok:
                    P = D + [0, 0, 0.04]; Pp = np.array([P[0], 0.17, P[2]]); Pz = np.array([P[0], 0.17, 0.125])
                    _, w = chain(ap, seg(P, Pp) + seg(Pp, Pz) + seg(Pz, [BOT[0], BOT[1], 0.125], polar=True) + seg([BOT[0], BOT[1], 0.125], BOT + [0, 0, 0.04]) + seg(BOT + [0, 0, 0.04], BOT - [0, 0, 0.02]))
                    route = ' route ' + ('OK' if w == 'ok' else w.split(' [')[0].split(' at')[0])
                out.append(f'{m*100:.0f}cm ' + ('grasp Y' if ok else '-') + route)
            print(f'yaw {yaw:+.0f} shelf {ox:+.2f},{oy:+.2f} D11 ({x:+.3f},{y:+.3f}) r {r*100:.1f}: ' + ' | '.join(out), flush=True)
            break
finally:
    L.v.update(orig); apply_shelf(0.01); print('scene restored')
