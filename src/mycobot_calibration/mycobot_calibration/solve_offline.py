"""Re-solve a saved run without the robot.

  ros2 run mycobot_calibration solve_offline ~/handeye_calib/20261104_153000
"""
import argparse
import os

import numpy as np
import yaml

from mycobot_calibration import solver
from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.geometry import T_to_xyzrpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--board', help='board yaml (default: the one recorded in samples.yaml)')
    ap.add_argument('--drop', type=int, nargs='*', default=[], help='sample indices to ignore')
    a = ap.parse_args()
    with open(os.path.join(os.path.expanduser(a.run_dir), 'samples.yaml')) as f:
        data = yaml.safe_load(f)
    target = CharucoTarget(a.board or data['board_file'])
    K, D = np.array(data['K']), np.array(data['D'])
    samples = [{
        'T_base_link6': np.array(s['T_base_link6']), 'T_cam_board': np.array(s['T_cam_board']),
        'ids': np.array(s['ids']), 'pixels': np.array(s['pixels']), 'K': K, 'D': D}
        for i, s in enumerate(data['samples']) if i not in a.drop]
    r = solver.calibrate(samples, target.corners3d)
    v = T_to_xyzrpy(r['X'])
    print(f"method {r['method']}, {r['n_samples']} samples")
    print(f"link6 -> camera_optical xyz {np.round(v[:3], 5).tolist()} "
          f"rpy {np.round(v[3:], 5).tolist()}")
    print(f"board scatter {r['board_trans_rms_mm']:.2f} mm / {r['board_rot_rms_deg']:.3f} deg, "
          f"chain reprojection {r['chain_reproj_rms_px']:.2f} px")
    worst = np.argsort(r['board_trans_per_sample_mm'])[::-1][:3]
    print('largest per-sample scatter [mm]:',
          {int(i): round(r['board_trans_per_sample_mm'][i], 2) for i in worst})
    for name, c in r['candidates'].items():
        print(f"  {name:10s} scatter {c['board_trans_rms_mm']:.2f} mm")


if __name__ == '__main__':
    main()
