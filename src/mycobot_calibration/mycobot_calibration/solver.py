"""Eye-in-hand calibration: X = T(link6 -> camera_optical).

1. Closed-form AX = XB with OpenCV calibrateHandEye (Tsai, Park, Horaud, Andreff,
   Daniilidis). The best one (smallest base->board scatter) is the initial guess.
2. Non-linear refinement: jointly optimise X and T(base -> board) to minimise the
   pixel reprojection error of every detected ChArUco corner in every sample
   (scipy least_squares, Huber loss). This uses the raw measurements instead of
   PnP poses, so it is the final answer.

Each sample: T_base_link6 (from TF / joint states), corner ids + pixels, K, D.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from mycobot_calibration.geometry import (
    inv, make_T, rot_angle, rvec_tvec_to_T, T_to_rvec_tvec)

METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def closed_form(T_base_link6, T_cam_board):
    """Run every OpenCV method. Returns {name: X}."""
    Rg = [T[:3, :3] for T in T_base_link6]
    tg = [T[:3, 3] for T in T_base_link6]
    Rt = [T[:3, :3] for T in T_cam_board]
    tt = [T[:3, 3] for T in T_cam_board]
    out = {}
    for name, m in METHODS.items():
        try:
            R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
            if np.all(np.isfinite(R)) and np.all(np.isfinite(t)):
                out[name] = make_T(R, t.reshape(3))
        except cv2.error:
            pass
    return out


def board_scatter(X, T_base_link6, T_cam_board):
    """Consistency check: T_base_board computed from every sample should be identical.

    Returns (T_base_board mean, translation std [m], rotation deviation [rad] list).
    """
    Tb = [Tg @ X @ Tc for Tg, Tc in zip(T_base_link6, T_cam_board)]
    t = np.array([T[:3, 3] for T in Tb])
    # rotation mean via SVD of the summed matrices
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Tb))
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    rot_dev = [rot_angle(Rm.T @ T[:3, :3]) for T in Tb]
    trans_dev = np.linalg.norm(t - t.mean(0), axis=1)
    return make_T(Rm, t.mean(0)), trans_dev, np.array(rot_dev)


def _pack(T):
    r, t = T_to_rvec_tvec(T)
    return np.concatenate([r, t])


def _unpack(v):
    return rvec_tvec_to_T(v[:3], v[3:6])


def reprojection_residuals(X, T_base_board, samples, obj_pts):
    res = []
    for s in samples:
        T_cb = inv(X) @ inv(s['T_base_link6']) @ T_base_board
        r, t = T_to_rvec_tvec(T_cb)
        proj, _ = cv2.projectPoints(obj_pts[s['ids']], r, t, s['K'], s['D'])
        res.append((proj.reshape(-1, 2) - s['pixels']).ravel())
    return np.concatenate(res)


def refine(X0, T_base_board0, samples, obj_pts):
    def f(v):
        return reprojection_residuals(_unpack(v[:6]), _unpack(v[6:]), samples, obj_pts)

    v0 = np.concatenate([_pack(X0), _pack(T_base_board0)])
    sol = least_squares(f, v0, loss='huber', f_scale=1.0, x_scale='jac')
    X, Tbb = _unpack(sol.x[:6]), _unpack(sol.x[6:])
    r = f(sol.x).reshape(-1, 2)
    # 1-sigma uncertainty of X from the Jacobian (Gauss-Newton approximation)
    try:
        J = sol.jac
        dof = max(1, r.size - 12)
        sigma2 = float(np.sum(r ** 2) / dof)
        cov = np.linalg.inv(J.T @ J) * sigma2
        std = np.sqrt(np.diag(cov))[:6]   # rvec(3) rad, t(3) m
    except np.linalg.LinAlgError:
        std = np.full(6, np.nan)
    return X, Tbb, r, std


def calibrate(samples, obj_pts):
    """Full pipeline. samples: list of dict(T_base_link6, T_cam_board, ids, pixels, K, D)."""
    if len(samples) < 3:
        raise ValueError('need at least 3 samples (15+ recommended)')
    Tg = [s['T_base_link6'] for s in samples]
    Tc = [s['T_cam_board'] for s in samples]
    cf = closed_form(Tg, Tc)
    if not cf:
        raise RuntimeError('all closed-form methods failed (not enough rotation variety?)')
    report = {'candidates': {}}

    def evaluate(name, X):
        _, td, rd = board_scatter(X, Tg, Tc)
        t_rms, r_rms = float(np.sqrt(np.mean(td ** 2))), float(np.sqrt(np.mean(rd ** 2)))
        report['candidates'][name] = {
            'X': X, 'board_trans_rms_mm': t_rms * 1000, 'board_rot_rms_deg': np.degrees(r_rms)}
        return t_rms + 0.1 * r_rms        # 1 deg ~ 1.7 mm

    scores = {name: evaluate(name, X) for name, X in cf.items()}
    best_cf = min(scores, key=scores.get)
    Tbb0, _, _ = board_scatter(cf[best_cf], Tg, Tc)
    # Reprojection refinement assumes exact robot kinematics. On a small arm with
    # ~0.1 deg joint error it can be worse than AX=XB, so it only wins if it also
    # reduces the base->board scatter (the independent consistency metric).
    X_ref, _, _, std = refine(cf[best_cf], Tbb0, samples, obj_pts)
    scores['REFINED'] = evaluate('REFINED', X_ref)
    method = min(scores, key=scores.get)
    X = report['candidates'][method]['X']

    T_base_board, td, rd = board_scatter(X, Tg, Tc)
    resid = reprojection_residuals(X, T_base_board, samples, obj_pts).reshape(-1, 2)
    per_sample, k = [], 0
    for s in samples:
        n = len(s['ids'])
        per_sample.append(float(np.sqrt(np.mean(np.sum(resid[k:k + n] ** 2, axis=1)))))
        k += n
    report.update({
        'method': method,
        'X': X,
        'T_base_board': T_base_board,
        'chain_reproj_rms_px': float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1)))),
        'chain_reproj_per_sample_px': per_sample,
        'board_trans_rms_mm': float(np.sqrt(np.mean(td ** 2)) * 1000),
        'board_trans_per_sample_mm': [float(v * 1000) for v in td],
        'board_rot_rms_deg': float(np.degrees(np.sqrt(np.mean(rd ** 2)))),
        'refined_X_std_rot_deg': [float(np.degrees(v)) for v in std[:3]],
        'refined_X_std_trans_mm': [float(v * 1000) for v in std[3:]],
        'n_samples': len(samples),
    })
    return report


def motion_diversity(T_base_link6):
    """Max / median relative rotation [deg] between sample pairs (want > 30 deg, many axes)."""
    angs = []
    for i in range(len(T_base_link6)):
        for j in range(i + 1, len(T_base_link6)):
            angs.append(np.degrees(rot_angle(T_base_link6[i][:3, :3].T @ T_base_link6[j][:3, :3])))
    return (float(np.max(angs)), float(np.median(angs))) if angs else (0.0, 0.0)
