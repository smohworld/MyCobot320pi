import math, numpy as np
from route import ik_any, contacts
B = np.array([0.2155, -0.0449, -0.208])
for flip in (False, True):
    for yd in (90, 270):
        r = [ik_any(B + [0, 0, h], math.radians(yd), flip) is not None for h in (-0.02, 0.04, 0.08)]
        print(f"{'reversed' if flip else 'normal  '} tool yaw {yd}: grasp/+4/+8 =", r, flush=True)
