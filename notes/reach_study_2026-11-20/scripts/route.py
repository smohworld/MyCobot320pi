import math, numpy as np
from scipy.optimize import least_squares
from scanlib import valid, sv, _wait
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from mycobot_calibration.handeye_node import ARM_JOINTS
def contacts(q):
    rq = GetStateValidity.Request(); rq.group_name = 'arm'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q]))
    return sorted({(c.contact_body_1, c.contact_body_2) for c in _wait(sv.call_async(rq), 20.0).contacts})
from fastik import arm, LO, HI, DOWN
from mycobot_calibration.geometry import rot_z
rng = np.random.default_rng(3)
def ik_local(p, yaw, q0):
    R = rot_z(yaw) @ DOWN
    f = lambda q: np.r_[(arm.fk(q)[:3, 3] - p) * 10, (arm.fk(q)[:3, :3].T @ R - np.eye(3)).ravel() * 0.3]
    r = least_squares(f, q0, bounds=(LO, HI))
    return r.x if np.linalg.norm(r.fun) < 3e-3 else None
def ik_any(p, yaw, flip):
    b = math.atan2(p[1], p[0]) + (math.pi if flip else 0)
    for _ in range(40):
        q0 = rng.uniform(LO, HI); q0[0] = math.remainder(b + rng.normal(0, 0.3), 2*math.pi)
        q = ik_local(p, yaw, np.clip(q0, LO, HI))
        if q is not None and abs(math.remainder(q[0]-b, 2*math.pi)) < math.radians(60) and valid(q): return q
D11 = (0.003, 0.313, 0.029); BOT = (0.2155, -0.0449, -0.208)
zc = D11[2] + 0.04
def route(r_mid, yaw_mode, y0=0.0, dx=0.0, dz=0.0):
    pts = []
    b0, b1 = math.atan2(D11[1], D11[0]), math.atan2(BOT[1], BOT[0])
    r0, r1 = math.hypot(*D11[:2]), math.hypot(*BOT[:2])
    x0, y0_ = D11[0], D11[1]
    pre = [(x0 + dx * t, y0_, zc + dz * t) for t in np.linspace(0, 1, 8)]          # sideways / up first
    pull = [(x0 + dx, y0_ + (r_mid - r0) * t, zc + dz) for t in np.linspace(0, 1, 45)[1:]]
    xyz = pre + pull
    for x, y, z in xyz: pts.append((math.hypot(x, y), math.atan2(y, x), z))
    b0 = math.atan2(y0_ + r_mid - r0, x0 + dx); r_mid = math.hypot(x0 + dx, y0_ + r_mid - r0)
    for t in np.linspace(0, 1, 80)[1:]: pts.append((r_mid, b0 + (b1 - b0) * t, zc + dz))           # swing
    for t in np.linspace(0, 1, 10)[1:]: pts.append((r_mid + (r1 - r_mid) * t, b1, zc + dz))        # out to the box
    for t in np.linspace(0, 1, 25)[1:]: pts.append((r1, b1, zc + dz + (BOT[2] + 0.04 - zc - dz) * t))   # down to 4 cm over the bot handle
    out = []
    for r, b, z in pts:
        yaw = y0 + math.pi + math.pi / 2 + (0 if yaw_mode == 'fixed' else (b - math.atan2(D11[1], D11[0])))
        out.append((np.array([r * math.cos(b), r * math.sin(b), z]), yaw))
    return out
if __name__ == "__main__":
    for flip, y0 in ((True, 0.0),):
      q = ik_any(np.array([D11[0], D11[1], zc]), y0 + math.pi + math.pi / 2, flip)
      print(f"start at D11 above, {'reversed' if flip else 'normal'} arm, tool yaw +{math.degrees(y0):.0f}:", None if q is None else np.round(np.degrees(q)).astype(int))
      if q is None: continue
      for r_mid, dx, dz in ((0.16, 0.02, 0.0), (0.13, 0.02, 0.0), (0.10, 0.02, 0.0), (0.13, 0.03, 0.0)):
          for ym in ('turn',):
              qq, worst, fail = q.copy(), 0.0, None
              for i, (p, yaw) in enumerate(route(r_mid, ym, y0, dx, dz)):
                  qn = ik_local(p, yaw, qq)
                  if qn is None: fail = f'no IK at step {i} {np.round(p, 3)}'; break
                  step = np.degrees(np.max(np.abs(qn - qq))); worst = max(worst, step)
                  if step > 12: fail = f'branch jump {step:.0f} deg at step {i} {np.round(p, 3)}'; break
                  if not valid(qn): fail = f'collision at step {i} {np.round(p, 3)} q {np.round(np.degrees(qn)).astype(int)} {contacts(qn)}'; break
                  qq = qn
              print(f'   dx {dx*100:+.0f} cm, pull to y {D11[1]+r_mid-0.313:.2f}: ' + (fail or f'OK, max step {worst:.1f} deg, end q {np.round(np.degrees(qq)).astype(int)}'), flush=True)
