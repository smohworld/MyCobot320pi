"""Eye-in-hand solver: X = T(link6 -> camera_optical_frame).

1. AX = XB closed form with every OpenCV cv2.calibrateHandEye method.
2. Reprojection refinement (scipy least_squares, Huber): X and T(base->board) together,
   minimising the pixel error of every detected corner in every sample.
3. The candidate with the smallest base->board scatter wins (if the robot kinematics
   are slightly off, the refinement is not always better, so it has to earn it).

sample = dict(T_base_link6, T_cam_board, ids, pixels, K, D)
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from mycobot_calibration.geometry import (
    T_to_rvec_tvec, inv, make_T, rot_angle, rvec_tvec_to_T)

METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def closed_form(Tg, Tc):
    out = {}
    args = ([T[:3, :3] for T in Tg], [T[:3, 3] for T in Tg],
            [T[:3, :3] for T in Tc], [T[:3, 3] for T in Tc])
    for name, m in METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(*args, method=m)
            if np.all(np.isfinite(R)) and np.all(np.isfinite(t)) and abs(np.linalg.det(R) - 1) < 1e-3:
                out[name] = make_T(R, t.reshape(3))
        except cv2.error:
            pass
    return out


def board_scatter(X, Tg, Tc):
    """T_base_board seen through every sample; returns (mean T, trans dev [m], rot dev [rad])."""
    Tb = [g @ X @ c for g, c in zip(Tg, Tc)]
    t = np.array([T[:3, 3] for T in Tb])
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Tb))
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return (make_T(Rm, t.mean(0)), np.linalg.norm(t - t.mean(0), axis=1),
            np.array([rot_angle(Rm.T @ T[:3, :3]) for T in Tb]))


def _pack(T):
    return np.concatenate(T_to_rvec_tvec(T))


def _unpack(v):
    return rvec_tvec_to_T(v[:3], v[3:6])


def reprojection_residuals(X, T_base_board, samples, obj):
    res = []
    for s in samples:
        r, t = T_to_rvec_tvec(inv(X) @ inv(s['T_base_link6']) @ T_base_board)
        p, _ = cv2.projectPoints(obj[s['ids']], r, t, s['K'], s['D'])
        res.append((p.reshape(-1, 2) - s['pixels']).ravel())
    return np.concatenate(res)


def refine(X0, Tbb0, samples, obj):
    def f(v):
        return reprojection_residuals(_unpack(v[:6]), _unpack(v[6:]), samples, obj)
    sol = least_squares(f, np.concatenate([_pack(X0), _pack(Tbb0)]),
                        loss='huber', f_scale=1.0, x_scale='jac')
    return _unpack(sol.x[:6]), _unpack(sol.x[6:])


def motion_diversity(Tg):
    """(max, median) relative rotation [deg] between flange poses. Want max > 40 deg."""
    a = [np.degrees(rot_angle(Tg[i][:3, :3].T @ Tg[j][:3, :3]))
         for i in range(len(Tg)) for j in range(i + 1, len(Tg))]
    return (float(np.max(a)), float(np.median(a))) if a else (0.0, 0.0)


def calibrate(samples, obj):
    if len(samples) < 3:
        raise ValueError('need at least 3 samples (15+ recommended)')
    Tg = [s['T_base_link6'] for s in samples]
    Tc = [s['T_cam_board'] for s in samples]
    cands = closed_form(Tg, Tc)
    if not cands:
        raise RuntimeError('every closed-form method failed (too little rotation between poses?)')

    report = {'candidates': {}}

    def score(name, X):
        _, td, rd = board_scatter(X, Tg, Tc)
        t_rms, r_rms = float(np.sqrt(np.mean(td ** 2))), float(np.sqrt(np.mean(rd ** 2)))
        report['candidates'][name] = {'X': X, 'board_trans_rms_mm': t_rms * 1000,
                                      'board_rot_rms_deg': float(np.degrees(r_rms))}
        return t_rms + 0.1 * r_rms            # 1 deg ~ 1.7 mm

    scores = {n: score(n, X) for n, X in cands.items()}
    best = min(scores, key=scores.get)
    Tbb0, _, _ = board_scatter(cands[best], Tg, Tc)
    try:
        X_ref, _ = refine(cands[best], Tbb0, samples, obj)
        scores['REFINED'] = score('REFINED', X_ref)
    except Exception:  # noqa: BLE001
        pass
    method = min(scores, key=scores.get)
    X = report['candidates'][method]['X']

    T_base_board, td, rd = board_scatter(X, Tg, Tc)
    r = reprojection_residuals(X, T_base_board, samples, obj).reshape(-1, 2)
    per, k = [], 0
    for s in samples:
        n = len(s['ids'])
        per.append(float(np.sqrt(np.mean(np.sum(r[k:k + n] ** 2, axis=1)))))
        k += n
    report.update({
        'method': method, 'X': X, 'T_base_board': T_base_board, 'n_samples': len(samples),
        'board_trans_rms_mm': float(np.sqrt(np.mean(td ** 2)) * 1000),
        'board_rot_rms_deg': float(np.degrees(np.sqrt(np.mean(rd ** 2)))),
        'board_trans_per_sample_mm': [float(v * 1000) for v in td],
        'chain_reproj_rms_px': float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))),
        'chain_reproj_per_sample_px': per,
    })
    return report
