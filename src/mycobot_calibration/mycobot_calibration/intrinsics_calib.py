"""Camera intrinsic calibration with the ChArUco board (OpenCV), no ROS needed.

Live (opens the camera directly; stop the ROS camera node first):
  ros2 run mycobot_calibration intrinsics_calib --device 0 --width 640 --height 480 \
      --board ~/final_project_ws/calibration/charuco_board.yaml --out ~/camera_calib

  keys:  SPACE capture   a auto-capture on/off   u undo last   c calibrate + save   q quit

From saved images (e.g. a previous session's ~/camera_calib/images):
  ros2 run mycobot_calibration intrinsics_calib --images ~/camera_calib/images \
      --board ... --out ~/camera_calib

Output: <out>/camera.yaml (ROS camera_info format, read by the camera and hand-eye nodes),
<out>/images/*.png (every capture), <out>/report.txt.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from mycobot_calibration.charuco import CharucoTarget
from mycobot_calibration.intrinsics import save_intrinsics


def detect(target, gray):
    corners, ids, _, _ = target.detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    # board.matchImagePoints() in OpenCV 4.7 reads these as ArUco marker corners and
    # mismatches them; index the chessboard corners by their ChArUco id instead
    obj = target.corners3d[ids.reshape(-1)].astype(np.float32)
    img = corners
    if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
        return None                     # one row of corners: no information on the plane
    return obj, img.reshape(-1, 2).astype(np.float32), corners, ids


F_GUESS = 470.0          # this 640x480 camera: ~470 px. Only a start value / aspect reference.
MIN_TILT_DEG = 12.0      # a view counts as "tilted" above this angle between board normal and ray
MIN_TILTED_VIEWS = 8


def tilt_deg(obj, img, size):
    """Angle (deg) between the board normal and the camera axis, via PnP with a rough K."""
    K0 = np.array([[F_GUESS, 0, size[0] / 2], [0, F_GUESS, size[1] / 2], [0, 0, 1]])
    ok, rvec, _ = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K0, None,
                               flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return 0.0
    nz = abs(cv2.Rodrigues(rvec)[0][2, 2])
    return float(np.degrees(np.arccos(np.clip(nz, 0, 1))))


def calibrate(views, size, fix_k3=True):
    """views: list of (obj, img). Drops views whose error is > 3x the median, once.
    Pixels are square, so fx/fy is held at the guess ratio (1.0); a start value keeps the
    solver from wandering when the views are only weakly informative."""
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    if fix_k3:
        flags |= cv2.CALIB_FIX_K3
    K0 = np.array([[F_GUESS, 0, size[0] / 2], [0, F_GUESS, size[1] / 2], [0, 0, 1]])
    objs, imgs = [v[0] for v in views], [v[1] for v in views]
    rms, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, size, K0.copy(), None, flags=flags)
    per = per_view_errors(objs, imgs, K, D, rv, tv)
    keep = [i for i, e in enumerate(per) if e <= 3 * np.median(per)]
    dropped = [i for i in range(len(views)) if i not in keep]
    if dropped and len(keep) >= 10:
        objs, imgs = [objs[i] for i in keep], [imgs[i] for i in keep]
        rms, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, size, K0.copy(), None, flags=flags)
        per = per_view_errors(objs, imgs, K, D, rv, tv)
    else:
        dropped = []
    return rms, K, D, per, dropped


def per_view_errors(objs, imgs, K, D, rv, tv):
    out = []
    for o, i, r, t in zip(objs, imgs, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, D)
        out.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
    return out


def coverage(views, size, grid=(8, 6)):
    """Fraction of image cells that contain at least one detected corner."""
    w, h = size
    hit = np.zeros(grid[::-1], bool)
    for _, img in views:
        gx = np.clip((img[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
        gy = np.clip((img[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def report_and_save(out, views, size, names, force=False):
    if len(views) < 10:
        print(f'need at least 10 views (have {len(views)}), 20-40 recommended')
        return False
    rms, K, D, per, dropped = calibrate(views, size)
    cov = coverage(views, size)
    tilts = np.array([tilt_deg(o, i, size) for o, i in views])
    n_tilt = int(np.sum(tilts >= MIN_TILT_DEG))
    lines = [
        f'board tilt: median {np.median(tilts):.0f} deg, max {tilts.max():.0f} deg, '
        f'{n_tilt} views >= {MIN_TILT_DEG:.0f} deg (need {MIN_TILTED_VIEWS})',
        f'views used: {len(views) - len(dropped)} / {len(views)}'
        + (f'  (dropped as outliers: {[names[i] for i in dropped]})' if dropped else ''),
        f'image size: {size[0]} x {size[1]}',
        f'reprojection rms: {rms:.3f} px   (good < 0.5, usable < 1.0)',
        f'per-view error: median {np.median(per):.3f}, max {np.max(per):.3f} px',
        f'image coverage: {cov * 100:.0f}% of an 8x6 grid   (want > 80%: also the corners/edges)',
        f'fx {K[0, 0]:.2f}  fy {K[1, 1]:.2f}  cx {K[0, 2]:.2f}  cy {K[1, 2]:.2f}',
        f'dist (k1 k2 p1 p2 k3): {np.round(D.ravel(), 5).tolist()}',
    ]
    warn = []
    bad = []
    if n_tilt < MIN_TILTED_VIEWS:
        bad.append(f'only {n_tilt} tilted views: focal length is NOT determined by flat views '
                   f'(fx/fy blow up). Tilt the board 20-40 deg (both directions), vary distance')
    if not (0.5 * size[0] < K[0, 0] < 1.6 * size[0]):
        bad.append(f'fx {K[0, 0]:.0f} is implausible for a {size[0]} px wide image')
    if rms > 1.0:
        warn.append('rms > 1 px: blurred / moving captures or wrong board size in the yaml')
    if cov < 0.8:
        warn.append('board did not reach the image corners/edges: distortion is poorly constrained')
    if abs(K[0, 2] - size[0] / 2) > 0.1 * size[0] or abs(K[1, 2] - size[1] / 2) > 0.1 * size[1]:
        warn.append('principal point far from the image centre: add tilted views')
    for ln in lines + ['WARNING: ' + w for w in warn] + ['REJECTED: ' + b for b in bad]:
        print(ln)
    if bad and not force:
        print('NOT saved (add more tilted views, or force with --force)')
        return False
    path = os.path.join(out, 'camera.yaml')
    save_intrinsics(path, K, D, size, rms)
    with open(os.path.join(out, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines + ['WARNING: ' + w for w in warn]) + '\n')
    print(f'saved {path}')
    return True


def from_images(args, target):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.images), '*.png'))
                   + glob.glob(os.path.join(os.path.expanduser(args.images), '*.jpg')))
    views, names, size = [], [], None
    for fn in files:
        gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        size = size or gray.shape[::-1]
        if gray.shape[::-1] != size:
            print(f'{fn}: different image size, skipped')
            continue
        d = detect(target, gray)
        if d is None:
            print(f'{os.path.basename(fn)}: board not found')
            continue
        views.append((d[0], d[1]))
        names.append(os.path.basename(fn))
    print(f'{len(views)} / {len(files)} images with the board')
    return report_and_save(args.out, views, size, names, args.force) if size else False


def live(args, target):
    dev = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # focus changes the intrinsics
    if not cap.isOpened():
        raise SystemExit(f'cannot open camera {args.device} (is the ROS camera node running?)')
    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    views, names, size = [], [], None
    auto, last_auto, last_corners = False, 0.0, None
    prev_img = None
    tilts = []
    print(__doc__.split('\n\n')[1])
    while True:
        ok, frame = cap.read()
        if not ok:
            print('frame grab failed')
            time.sleep(0.1)
            continue
        size = frame.shape[1::-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = detect(target, gray)
        vis = frame.copy()
        for _, img in views:                        # coverage so far
            for x, y in img[::2]:
                cv2.circle(vis, (int(x), int(y)), 2, (255, 160, 0), -1)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        diff = float(np.mean(cv2.absdiff(gray, prev_img))) if prev_img is not None else 99.0
        prev_img = gray
        if d is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, d[2], d[3], (0, 255, 0))
        n = 0 if d is None else len(d[0])
        tilt = tilt_deg(d[0], d[1], size) if d is not None else 0.0
        n_tilted = sum(1 for t in tilts if t >= MIN_TILT_DEG)

        # auto-capture: first failing condition is shown on screen
        wait = 1.0 - (time.monotonic() - last_auto)
        shift = None
        if d is not None and last_corners is not None and len(last_corners) == len(d[1]):
            shift = float(np.median(np.linalg.norm(last_corners - d[1], axis=1)))
        if n < args.min_corners:
            reason = f'too few corners ({n} < {args.min_corners})'
        elif sharp < args.min_sharpness:
            reason = f'blurred (sharp {sharp:.0f} < {args.min_sharpness:.0f})'
        elif diff >= args.still_threshold:
            reason = f'hold still (motion {diff:.1f} >= {args.still_threshold:.1f})'
        elif wait > 0:
            reason = f'cooldown {wait:.1f}s'
        elif tilt < MIN_TILT_DEG and len(tilts) - n_tilted >= 4:
            reason = f'tilt the board! (tilt {tilt:.0f} < {MIN_TILT_DEG:.0f} deg)'
        elif shift is not None and shift <= args.min_move_px:
            reason = f'move to a new spot (moved {shift:.0f} <= {args.min_move_px:.0f}px)'
        else:
            reason = None
        auto_ok = auto and reason is None

        txt = (f'views {len(views)}  corners {n}/{target.n_corners}  '
               f'cov {coverage(views, size) * 100:.0f}%  tilt {tilt:.0f}deg  '
               f'tilted {n_tilted}/{MIN_TILTED_VIEWS}  auto {"ON" if auto else "off"}')
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if auto:
            st, col = ('AUTO: capturing', (0, 255, 0)) if reason is None else \
                ('AUTO: ' + reason, (0, 0, 255))
            cv2.putText(vis, st, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(vis, 'SPACE capture  a auto  u undo  c calibrate  q quit', (10, size[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('intrinsics_calib', vis)
        key = cv2.waitKey(1) & 0xFF

        want = key == ord(' ') or auto_ok
        if want:
            if d is None or n < args.min_corners:
                print(f'not captured: {n} corners (need {args.min_corners})')
            elif sharp < args.min_sharpness:
                print(f'not captured: blurred (sharpness {sharp:.0f} < {args.min_sharpness})')
            else:
                name = f'{len(views):03d}.png'
                cv2.imwrite(os.path.join(img_dir, name), frame)
                views.append((d[0], d[1]))
                tilts.append(tilt)
                names.append(name)
                last_corners, last_auto = d[1].copy(), time.monotonic()
                print(f'captured {name}: {n} corners')
        elif key == ord('a'):
            auto = not auto
        elif key == ord('u') and views:
            views.pop()
            tilts.pop()
            os.remove(os.path.join(img_dir, names.pop()))
            last_corners = views[-1][1] if views else None
            print(f'removed last view, {len(views)} left')
        elif key == ord('c'):
            report_and_save(args.out, views, size, names, args.force)
        elif key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    from ament_index_python.packages import get_package_share_directory
    default_board = os.path.join(get_package_share_directory('mycobot_calibration'),
                                 'config', 'charuco_board.yaml')
    p = argparse.ArgumentParser(description='ChArUco camera intrinsic calibration')
    p.add_argument('--board', default=default_board, help='charuco_board.yaml (measured sizes)')
    p.add_argument('--out', default='~/camera_calib', help='output folder')
    p.add_argument('--device', default='0', help='camera index or /dev/videoN')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--images', default='', help='calibrate from saved images instead of live')
    p.add_argument('--force', action='store_true', help='save even if the checks reject it')
    p.add_argument('--min-corners', type=int, default=12)
    p.add_argument('--min-sharpness', type=float, default=60.0, help='Laplacian variance')
    p.add_argument('--min-move-px', type=float, default=40.0, help='auto-capture: board moved')
    p.add_argument('--still-threshold', type=float, default=2.0, help='auto-capture: frame diff')
    args, _ = p.parse_known_args()            # ignore --ros-args from ros2 run
    args.out = os.path.expanduser(args.out)
    os.makedirs(args.out, exist_ok=True)
    target = CharucoTarget(os.path.expanduser(args.board))
    print(f'board: {target.describe()} ({args.board})')
    if args.images:
        from_images(args, target)
    else:
        live(args, target)


if __name__ == '__main__':
    main()
