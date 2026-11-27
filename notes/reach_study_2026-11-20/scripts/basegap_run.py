# tray_handle_pick with a report of the held tray's clearance to the arm base parts
# (base, link1, robot_mount) along every held path, after each relay dense check: per path
# the smallest gap (1 mm steps up to 50 mm) and where. Report only; lines start "BASEGAP".
#   ROS_DOMAIN_ID=77 python3 basegap_run.py <tray_handle_pick args>
import math, os, sys
import numpy as np
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.moveit_client import _wait
from mycobot_tasks.handle_pick import relay as R

orig_verify = R.RelayMixin.relay_verify
# GAP_PARTS=shelf_rack (comma list): the gap to these parts instead of the arm base parts
PARTS = set(os.environ['GAP_PARTS'].split(',')) if os.environ.get('GAP_PARTS') else R.BASE_PARTS


def base_hit(self, q, pad):
    rq = GetStateValidity.Request(group_name='arm')
    rq.robot_state = RobotState(
        joint_state=JointState(name=ARM_JOINTS + ['gripper_controller'],
                               position=[float(v) for v in q] + [float(self.args.hold_grip)]),
        attached_collision_objects=[self._tray_attached(pad, R.BASE_PAD_ID)], is_diff=False)
    r = _wait(self._rl_sv.call_async(rq), 20.0)
    parts = {b for c in r.contacts for b in (c.contact_body_1, c.contact_body_2)} & PARTS
    return sorted(parts)


def gap(self, q):
    if not base_hit(self, q, 0.05):
        return 0.05, []
    lo, hi = 0.0, 0.05                 # hit at hi, find the smallest pad that hits (1 mm)
    while hi - lo > 0.001:
        m = (lo + hi) / 2
        if base_hit(self, q, m):
            hi = m
        else:
            lo = m
    return lo, base_hit(self, q, hi)


def verify(self, steps, title, start_tray=True):
    near = orig_verify(self, steps, title, start_tray)
    held, prev = False, None
    for kind, data, (label, _) in steps:
        if kind == 'grasp':
            held = True
            continue
        if kind == 'release':
            held = False
            continue
        if kind != 'traj' or not data:
            continue
        if held:
            pts = ([prev] if prev is not None else []) + list(data)
            dense = [pts[0]]
            for u, v in zip(pts, pts[1:]):
                m = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(v - u)))))))
                dense += [u + (v - u) * t for t in np.linspace(0, 1, m + 1)[1:]]
            gs = [(gap(self, q), i) for i, q in enumerate(dense)]
            if os.environ.get('GAP_PROFILE') and label.startswith(os.environ['GAP_PROFILE']):
                for (g_, p_), i_ in gs[::3]:
                    T_ = self._rl_arm.fk(dense[i_])
                    print(f'GAPPROF {i_} tcp ({T_[0, 3] * 1000:.0f}, {T_[1, 3] * 1000:.0f}) '
                          f'gap {g_ * 1000:.0f} mm', flush=True)
            best = min(gs, key=lambda x: x[0][0])
            (g, parts), i = best
            T = self._rl_arm.fk(dense[i])
            print(f'BASEGAP {title} | {label}: min {g * 1000:.0f} mm{"+" if g >= 0.05 else ""} '
                  f'to {parts or "-"} at pose {i}/{len(dense)} tcp ({T[0, 3] * 1000:.0f}, '
                  f'{T[1, 3] * 1000:.0f}, {T[2, 3] * 1000:.0f}) mm', flush=True)
        prev = data[-1]
    return near


R.RelayMixin.relay_verify = verify
from mycobot_tasks.handle_pick.cli import main   # noqa: E402
sys.argv = ['tray_handle_pick'] + sys.argv[1:]
main()
