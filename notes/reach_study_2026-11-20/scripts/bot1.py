import math, numpy as np, rclpy
from fastik import solve
from diag import check
for name, p, yaw in (('bot', (0.2155, -0.0449, -0.208 + 0.04), math.pi), ('desk top', (0.2, -0.15, 0.05), 0.0)):
    q = solve(np.array(p), yaw); print(name, None if q is None else (np.round(np.degrees(q)).astype(int), check(q)))
