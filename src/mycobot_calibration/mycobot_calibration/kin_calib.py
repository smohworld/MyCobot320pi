"""Joint-offset calibration from hand-eye runs (samples.yaml): which constant angle error of
joints J2..J5 makes the chain  base -> FK(q + delta) -> X -> camera -> board  agree best with
the pixels of the ChArUco corners?

Why: the hand-eye result X only absorbs a constant camera mount error. A joint that reads a
degree off moves the camera by several mm, and the error changes with the pose, so a good X
still gives 4 mm scatter on new viewpoints (verify_views).

Model (all runs together): delta (rad, selected joints), X = T(link6 -> camera) and one board
pose per run (the board does not move within a run) are fitted to the corner pixels of every
sample (robust least squares). J1 and J6 are left out on purpose: a J1 offset is the same as
turning the board pose, a J6 offset the same as turning X, so they cannot be observed.

  python3 -m mycobot_calibration.kin_calib --urdf robot.urdf          # all ~/handeye_calib runs
  python3 -m mycobot_calibration.kin_calib --runs ~/handeye_calib/20261111_171752 ~/handeye_calib/20261111_113536

Checks printed: reprojection rms and board scatter before / after, a leave-one-run-out test
(fit on the other runs, board pose of the held-out run fitted alone: does delta help on data it
never saw?), the delta of every run fitted alone (do they agree?), and the new driver.yaml
joint_offsets_deg (= current + delta: q_true = q_read + delta, so the offset moves by delta).

The samples must come from runs made with the same driver joint_offsets_deg. The URDF is read
from the running /robot_state_publisher unless --urdf is given.
"""
import argparse
import glob
import os
import re
import subprocess
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import inv, make_T, rpy_to_R, rvec_tvec_to_T
from mycobot_calibration.intrinsics import load_intrinsics

ARM_JOINTS = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
              'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']


# ----------------------------------------------------------------- kinematics
def _floats(text, default):
    return [float(v) for v in (text or default).split()]


class Arm:
    """Forward kinematics base -> tip from the URDF text (revolute / fixed joints)."""

    def __init__(self, urdf_xml, base='base', tip='link6'):
        root = ET.fromstring(urdf_xml)
        child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link = [], tip
        while link != base:
            j = child[link]
            chain.append(j)
            link = j.find('parent').get('link')
        self.joints = []
        for j in reversed(chain):
            o = j.find('origin')
            xyz = _floats(o.get('xyz') if o is not None else None, '0 0 0')
            rpy = _floats(o.get('rpy') if o is not None else None, '0 0 0')
            a = j.find('axis')
            axis = None
            if j.get('type') in ('revolute', 'continuous'):
                axis = np.array(_floats(a.get('xyz') if a is not None else None, '0 0 1'))
                axis /= np.linalg.norm(axis)
            self.joints.append((j.get('name'), make_T(rpy_to_R(*rpy), xyz), axis))
        self.index = {n: i for i, n in enumerate(ARM_JOINTS)}

    def fk(self, q, delta=None):
        """q, delta: 6 values in ARM_JOINTS order."""
        T = np.eye(4)
        for name, T0, axis in self.joints:
            T = T @ T0
            if axis is not None:
                i = self.index[name]
                ang = q[i] + (0.0 if delta is None else delta[i])
                T = T @ make_T(cv2.Rodrigues(axis * ang)[0], np.zeros(3))
        return T


def get_urdf(path):
    if path:
        return open(os.path.expanduser(path)).read()
    try:
        out = subprocess.run(['ros2', 'param', 'get', '--hide-type', '/robot_state_publisher',
                              'robot_description'], capture_output=True, text=True, timeout=15)
        i = out.stdout.find('<robot')
        if i >= 0:
            return out.stdout[i:]
    except (OSError, subprocess.TimeoutExpired):
        pass
    raise SystemExit('no URDF: pass --urdf <file> (or run with the robot launch up)')


# ----------------------------------------------------------------------- data
def pnp(obj, pix, K, D):
    ok, rv, tv = cv2.solvePnP(obj, pix, K, D, flags=cv2.SOLVEPNP_IPPE)
    rv, tv = cv2.solvePnPRefineLM(obj, pix, K, D, rv, tv)
    return rvec_tvec_to_T(rv.reshape(3), tv.reshape(3))


def load_run(path, target, K, D):
    path = os.path.expanduser(path)
    s = yaml.safe_load(open(os.path.join(path, 'samples.yaml')))
    r = yaml.safe_load(open(os.path.join(path, 'result.yaml')))
    samples = []
    for x in s['samples']:
        if x.get('q') is None:
            continue
        ids = np.array(x['ids'], int)
        obj = target.corners3d[ids]
        pix = np.array(x['pixels'], float)
        samples.append(dict(q=np.array(x['q'], float), Tbl6=np.array(x['T_base_link6']),
                            obj=obj, pix=pix, Tcb=pnp(obj, pix, K, D)))
    return dict(name=os.path.basename(path.rstrip('/')), samples=samples,
                X0=np.array(r['link6_to_camera_optical_T']), Tbb0=np.array(r['T_base_board']))


def T_to_vec(T):
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].reshape(3), T[:3, 3]])


def vec_to_T(v):
    return make_T(cv2.Rodrigues(np.asarray(v[:3], float))[0], v[3:6])


# ------------------------------------------------------------------------ fit
class Fit:
    """Least squares over [delta(sel) | X | board of each run]; delta / X can be frozen."""

    def __init__(self, arm, K, D, sel):
        self.arm, self.K, self.D, self.sel = arm, K, D, list(sel)

    def delta_vec(self, d):
        out = np.zeros(6)
        out[self.sel] = d
        return out

    def residuals_run(self, run, delta, X, Tbb):
        res = []
        for s in run['samples']:
            Tcb = inv(self.arm.fk(s['q'], delta) @ X) @ Tbb
            proj, _ = cv2.projectPoints(s['obj'], cv2.Rodrigues(Tcb[:3, :3])[0], Tcb[:3, 3],
                                        self.K, self.D)
            res.append((proj.reshape(-1, 2) - s['pix']).ravel())
        return np.concatenate(res)

    def solve(self, runs, fit_delta=True, fit_X=True, delta0=None, X0=None):
        nd = len(self.sel) if fit_delta else 0
        d0 = np.zeros(len(self.sel)) if delta0 is None else np.asarray(delta0, float)[self.sel]
        X0 = runs[0]['X0'] if X0 is None else X0
        x0 = np.concatenate([d0 if fit_delta else [], T_to_vec(X0) if fit_X else [],
                             *[T_to_vec(r['Tbb0']) for r in runs]])

        def unpack(x):
            i = 0
            d = self.delta_vec(x[:nd]) if fit_delta else self.delta_vec(d0)
            i = nd
            X = vec_to_T(x[i:i + 6]) if fit_X else X0
            i += 6 if fit_X else 0
            return d, X, [vec_to_T(x[i + 6 * k:i + 6 * k + 6]) for k in range(len(runs))]

        def fun(x):
            d, X, B = unpack(x)
            return np.concatenate([self.residuals_run(r, d, X, b) for r, b in zip(runs, B)])

        sol = least_squares(fun, x0, loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=100)
        d, X, B = unpack(sol.x)
        res = fun(sol.x)
        n, m = len(sol.x), len(res)
        sigma2 = float(np.sum(res ** 2) / max(m - n, 1))
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * sigma2
        sd = np.sqrt(np.diag(cov)[:nd]) if fit_delta else np.zeros(0)
        return dict(delta=d, X=X, boards=B, rms=float(np.sqrt(np.mean(res ** 2))),
                    delta_sd=sd, n_res=m)

    def board_only(self, run, delta, X):
        """Fit just the board pose of one run (delta, X fixed); returns residual rms [px]."""
        sol = least_squares(lambda v: self.residuals_run(run, delta, X, vec_to_T(v)),
                            T_to_vec(run['Tbb0']), loss='soft_l1', f_scale=1.0, x_scale='jac')
        return float(np.sqrt(np.mean(sol.fun ** 2)))

    def scatter_mm(self, run, delta, X):
        """Board position scatter: T_base_board seen through every sample's PnP pose."""
        t = np.array([(self.arm.fk(s['q'], delta) @ X @ s['Tcb'])[:3, 3] for s in run['samples']])
        return float(np.sqrt(np.mean(np.sum((t - t.mean(0)) ** 2, axis=1))) * 1000.0)


# --------------------------------------------------------------------- report
def current_offsets(driver_yaml):
    try:
        m = re.search(r'joint_offsets_deg:\s*\[([^\]]*)\]', open(driver_yaml).read())
        return [float(v) for v in m.group(1).split(',')]
    except (OSError, AttributeError, ValueError):
        return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.dirname(here)
    ap = argparse.ArgumentParser(description='joint-offset calibration from hand-eye runs')
    ap.add_argument('--runs', nargs='*', help='hand-eye run folders (samples.yaml + result.yaml)')
    ap.add_argument('--urdf', default='', help='URDF file (default: /robot_state_publisher)')
    ap.add_argument('--intrinsics', default=os.path.join(pkg, 'config', 'camera.yaml'))
    ap.add_argument('--board', default=os.path.join(pkg, 'config', 'charuco_board.yaml'))
    ap.add_argument('--joints', default='2,3,4,5', help='joints to fit, 1-based (default 2,3,4,5)')
    ap.add_argument('--driver-yaml', default=os.path.join(
        os.path.dirname(pkg), 'mycobot_bridge', 'config', 'driver.yaml'))
    ap.add_argument('--out', default='', help='write the result as yaml')
    a, _ = ap.parse_known_args()

    runs_dirs = a.runs or sorted(d for d in glob.glob(os.path.expanduser('~/handeye_calib/2026*'))
                                 if os.path.exists(os.path.join(d, 'samples.yaml')))
    K, D, _ = load_intrinsics(a.intrinsics)
    target = CharucoTarget(a.board)
    arm = Arm(get_urdf(a.urdf))
    sel = [int(v) - 1 for v in a.joints.split(',')]
    runs = [load_run(d, target, K, D) for d in runs_dirs]
    print(f'intrinsics {a.intrinsics}: fx {K[0, 0]:.1f}; joints fitted: '
          f'{[f"J{j + 1}" for j in sel]}; runs: ' + ', '.join(
              f"{r['name']} ({len(r['samples'])})" for r in runs))

    # forward kinematics must reproduce the TF the samples were taken with
    err = [np.linalg.norm(arm.fk(s['q'])[:3, 3] - s['Tbl6'][:3, 3]) * 1000
           for r in runs for s in r['samples']]
    print(f'FK check vs recorded T_base_link6: max {max(err):.3f} mm, mean {np.mean(err):.3f} mm')
    if max(err) > 1.0:
        raise SystemExit('FK from the URDF does not match the samples: wrong URDF / '
                         'joint_offsets changed between runs?')

    fit = Fit(arm, K, D, sel)
    base = fit.solve(runs, fit_delta=False)          # X and boards only
    full = fit.solve(runs, fit_delta=True)
    print('\n== all runs together ==')
    print(f'reprojection rms: {base["rms"]:.3f} px (delta = 0)  ->  {full["rms"]:.3f} px (fitted)')
    print('delta [deg]: ' + ', '.join(
        f'J{j + 1} {np.degrees(full["delta"][j]):+.2f} +- {np.degrees(sd):.2f}'
        for j, sd in zip(sel, full['delta_sd'])))
    for r in runs:
        print(f"  {r['name']}: board scatter {fit.scatter_mm(r, np.zeros(6), base['X']):.2f} mm "
              f"-> {fit.scatter_mm(r, full['delta'], full['X']):.2f} mm")

    print('\n== leave one run out (fit on the others, test on the held-out run) ==')
    print('   reprojection rms of the held-out run, only its board pose fitted:')
    hold = []
    for k, r in enumerate(runs):
        others = runs[:k] + runs[k + 1:]
        if not others:
            continue
        b0 = fit.solve(others, fit_delta=False)
        b1 = fit.solve(others, fit_delta=True)
        e0 = fit.board_only(r, np.zeros(6), b0['X'])
        e1 = fit.board_only(r, b1['delta'], b1['X'])
        s0 = fit.scatter_mm(r, np.zeros(6), b0['X'])
        s1 = fit.scatter_mm(r, b1['delta'], b1['X'])
        hold.append((e0, e1, s0, s1))
        print(f"  {r['name']}: {e0:.3f} -> {e1:.3f} px   board scatter {s0:.2f} -> {s1:.2f} mm"
              f"   ({'better' if e1 < e0 else 'WORSE'})")
    if hold:
        h = np.array(hold)
        print(f'  mean: {h[:, 0].mean():.3f} -> {h[:, 1].mean():.3f} px, '
              f'scatter {h[:, 2].mean():.2f} -> {h[:, 3].mean():.2f} mm')

    if len(runs) > 1:
        print('\n== delta of every run fitted alone (agreement = the offsets are real) ==')
        for r in runs:
            one = fit.solve([r], fit_delta=True)
            print(f"  {r['name']}: " + ', '.join(
                f'J{j + 1} {np.degrees(one["delta"][j]):+.2f}' for j in sel) + ' deg')

    cur = current_offsets(a.driver_yaml)
    if cur is not None:
        new = [round(c + float(np.degrees(full['delta'][i])), 2) for i, c in enumerate(cur)]
        print(f'\ndriver.yaml joint_offsets_deg now {cur}\n  -> suggested {new}  '
              '(only apply if the leave-one-run-out test above says "better")')
    if a.out:
        with open(os.path.expanduser(a.out), 'w') as f:
            yaml.safe_dump({'joints': [j + 1 for j in sel],
                            'delta_deg': [float(np.degrees(full['delta'][j])) for j in sel],
                            'delta_sd_deg': [float(np.degrees(v)) for v in full['delta_sd']],
                            'rms_px_before': base['rms'], 'rms_px_after': full['rms'],
                            'runs': [r['name'] for r in runs]}, f)


if __name__ == '__main__':
    main()
