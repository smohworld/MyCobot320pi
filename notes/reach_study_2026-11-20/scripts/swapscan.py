import math, numpy as np
from mycobot_tasks import layout as lay, place_scene as ps
L = lay.Layout('~/.ros/layout.yaml')
OTH = ['desk', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box']
def hits(): 
    o, _ = ps.build(L.args()); return ps.box_hits(o, [k for k in o if k.startswith('shelf')], OTH), o
# TurtleBot on the +y side first (shelf off): smallest r
L.v['with_shelf'] = False
bots = []
for byaw in (90.0, -90.0, 0.0, 180.0):
    for bx in np.arange(-0.15, 0.151, 0.01):
        for by in np.arange(0.10, 0.40, 0.01):
            L.v['bot'] = [float(bx), float(by), byaw]
            o, _ = ps.build(L.args())
            if ps.box_hits(o, ['turtlebot', 'bot_box'], ['desk', 'desk_legs', 'robot_mount']): continue
            bots.append((math.hypot(bx, by), float(bx), float(by), byaw)); break
bots.sort(); print('TurtleBot on +y, best:', [tuple(round(v, 3) for v in b) for b in bots[:4]])
L.v['with_shelf'] = True
out = []
for r_b, bx, by, byaw in bots[:1]:
    L.v['bot'] = [bx, by, byaw]
    for yaw in (170.0, 180.0, 190.0, -10.0, 0.0, 10.0):
        for sx in (-0.097, 0.097):
            best = None
            for ox in np.arange(-0.2, 0.9, 0.01):
                for oy in np.arange(-0.7, 0.7, 0.01):
                    L.v.update(shelf=[float(ox), float(oy), yaw], shelf_x=sx)
                    x, y, z = L.fork_point()
                    if x < 0.12 or math.hypot(x, y) > 0.33 or (best and math.hypot(x, y) >= best[0]): continue
                    h, _ = hits()
                    if not h: best = (math.hypot(x, y), float(ox), float(oy), x, y)
            if best: out.append((yaw, sx) + best); print(f'shelf yaw {yaw:+.0f} shelf_x {sx:+.3f}: shelf {best[1]:+.2f},{best[2]:+.2f}  D11 ({best[3]:+.3f},{best[4]:+.3f}) r {best[0]*100:.1f} cm', flush=True)
