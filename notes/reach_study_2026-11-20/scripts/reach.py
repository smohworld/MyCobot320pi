import math, numpy as np
from fastik import solve
def rmax(z, bearing, yaw):
    lo, hi = 0.10, 0.45
    if solve(np.array([lo*math.cos(bearing), lo*math.sin(bearing), z]), yaw) is None: return None
    for _ in range(9):
        m = (lo+hi)/2
        if solve(np.array([m*math.cos(bearing), m*math.sin(bearing), z]), yaw) is not None: lo = m
        else: hi = m
    return lo
print('TurtleBot handle z -0.208, tray yaw 180:')
for d in (0.0, 0.02, 0.04, 0.06):
    x, y = 0.2155, -0.0449 - d
    for h in (0.02, 0.08):
        q = solve(np.array([x, y, -0.208 + h]), math.pi)
        print(f'  robot +y {d*100:.0f} cm  r {math.hypot(x,y)*100:.1f} cm  tip +{h*100:.0f} cm: {"OK" if q is not None else "no IK"}')
b = math.atan2(-0.0449, 0.2155)
for h in (0.02, 0.08):
    print(f'  max r at bot bearing, tip +{h*100:.0f}: {rmax(-0.208+h, b, math.pi)*100:.1f} cm')
print('shelf D11 (bearing 90, tray yaw 180), max r:')
for ft in (0.263, 0.287, 0.304):
    for h in (0.02, 0.08):
        r = rmax(-0.306 + ft + 0.072 + h, math.pi/2, math.pi)
        print(f'  fork_top {ft*1000:.0f}  tip +{h*100:.0f} cm: {r*100:.1f} cm' if r else '  none')
