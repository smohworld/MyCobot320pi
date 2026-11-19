#!/usr/bin/env python3
"""Standalone ChArUco camera intrinsic calibration (no ROS needed). Output: ROS camera_info yaml.

Needs: Python 3.8+, opencv-contrib-python >= 4.7, numpy, pyyaml
       (tested setup: Ubuntu 22.04 / Python 3.10 / opencv-contrib-python 4.10.0.84 / numpy 1.26.4)

Live, camera plugged into this computer (use the SAME resolution/format as on the robot):
  python3 charuco_intrinsics.py --device 0 --width 320 --height 240 --fourcc MJPG

Remote, camera stays on the robot (ROS 2 image topic; source ROS first, then the venv):
  source /opt/ros/humble/setup.bash && source venv/bin/activate
  python3 charuco_intrinsics.py --topic /image_raw          # or /image_raw/compressed

From a folder of saved images (*.png / *.jpg):
  python3 charuco_intrinsics.py --images ./calib_out/images

keys:  SPACE capture   a auto-capture on/off   u undo last   c calibrate + save
       v undistorted preview on/off (after c)   q quit

Output in --out (default ./calib_out): camera.yaml (ROS camera_info format, usable as
usb_cam camera_info_url), report.txt, images/*.png (every capture, re-usable with --images).
"""
import argparse
import datetime
import glob
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_TILT_DEG = 12.0      # a view counts as "tilted" above this angle (board normal vs camera axis)
MIN_TILTED_VIEWS = 8     # tilted views are what pins down the focal length
MIN_VIEWS = 10


# ----------------------------------------------------------------------------- board
class Board:

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            c = yaml.safe_load(f)
        if not hasattr(cv2.aruco, 'CharucoDetector'):
            sys.exit(f'OpenCV {cv2.__version__} is too old: need opencv-contrib-python >= 4.7')
        self.cols, self.rows = int(c['squares_x']), int(c['squares_y'])
        self.square, self.marker = float(c['square_length']), float(c['marker_length'])
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c['dictionary']))
        self.board = cv2.aruco.CharucoBoard((self.cols, self.rows), self.square, self.marker, d)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.corners3d = self.board.getChessboardCorners().reshape(-1, 3).astype(np.float32)
        self.n_corners = len(self.corners3d)
        self.name = (f'{self.cols}x{self.rows} squares, square {self.square * 1000:.2f} mm, '
                     f'marker {self.marker * 1000:.2f} mm, {c["dictionary"]}')

    def detect(self, gray):
        """(obj Nx3, img Nx2, corners, ids) or None."""
        corners, ids, _, _ = self.detector.detectBoard(gray)
        if ids is None or len(ids) < 6:
            return None
        obj = self.corners3d[ids.reshape(-1)]
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
            return None                                  # one row of corners only
        return obj, corners.reshape(-1, 2).astype(np.float32), corners, ids


# ----------------------------------------------------------------------------- math
def k_guess(size, f):
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], np.float64)


def tilt_deg(obj, img, size, f):
    ok, rvec, _ = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), k_guess(size, f),
                               None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(abs(cv2.Rodrigues(rvec)[0][2, 2]), 0, 1))))


def per_view_errors(objs, imgs, K, D, rv, tv):
    out = []
    for o, i, r, t in zip(objs, imgs, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, D)
        out.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
    return out


def calibrate(views, size, f):
    """Square pixels (fx = fy), k3 fixed; drops views with error > 3x median once."""
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3
    objs, imgs = [v[0] for v in views], [v[1] for v in views]
    rms, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, size, k_guess(size, f), None, flags=flags)
    per = per_view_errors(objs, imgs, K, D, rv, tv)
    keep = [i for i, e in enumerate(per) if e <= 3 * np.median(per)]
    dropped = [i for i in range(len(views)) if i not in keep]
    if dropped and len(keep) >= MIN_VIEWS:
        objs, imgs = [objs[i] for i in keep], [imgs[i] for i in keep]
        rms, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, size, k_guess(size, f), None,
                                                flags=flags)
        per = per_view_errors(objs, imgs, K, D, rv, tv)
    else:
        dropped = []
    return rms, K, D, per, dropped


def coverage(views, size, grid=(8, 6)):
    hit = np.zeros(grid[::-1], bool)
    for _, img in views:
        gx = np.clip((img[:, 0] / size[0] * grid[0]).astype(int), 0, grid[0] - 1)
        gy = np.clip((img[:, 1] / size[1] * grid[1]).astype(int), 0, grid[1] - 1)
        hit[gy, gx] = True
    return float(hit.mean())


# ----------------------------------------------------------------------------- output
def save_ros_yaml(path, K, D, size, camera_name, comment):
    d = [float(x) for x in D.ravel()[:5]] + [0.0] * max(0, 5 - D.size)
    P = np.zeros((3, 4))
    P[:3, :3] = K
    data = {
        'image_width': int(size[0]), 'image_height': int(size[1]), 'camera_name': camera_name,
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(x) for x in K.ravel()]},
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': d},
        'rectification_matrix': {'rows': 3, 'cols': 3,
                                 'data': [float(x) for x in np.eye(3).ravel()]},
        'projection_matrix': {'rows': 3, 'cols': 4, 'data': [float(x) for x in P.ravel()]},
    }
    with open(path, 'w') as f:
        f.write(''.join(f'# {ln}\n' for ln in comment))
        yaml.safe_dump(data, f, sort_keys=False)


def report_and_save(a, views, size, names):
    """Returns (K, D) if saved, else None."""
    if len(views) < MIN_VIEWS:
        print(f'need at least {MIN_VIEWS} views (have {len(views)}), 25-40 recommended')
        return None
    rms, K, D, per, dropped = calibrate(views, size, a.f_guess)
    cov = coverage(views, size)
    tilts = np.array([tilt_deg(o, i, size, a.f_guess) for o, i in views])
    n_tilt = int(np.sum(tilts >= MIN_TILT_DEG))
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    lines = [
        f'{datetime.datetime.now():%Y-%m-%d %H:%M}  OpenCV {cv2.__version__}',
        f'board: {a.board_obj.name}',
        f'image size: {size[0]} x {size[1]}',
        f'views used: {len(views) - len(dropped)} / {len(views)}'
        + (f'  (dropped outliers: {[names[i] for i in dropped]})' if dropped else ''),
        f'board tilt: median {np.median(tilts):.0f} deg, max {tilts.max():.0f} deg, '
        f'{n_tilt} views >= {MIN_TILT_DEG:.0f} deg (need {MIN_TILTED_VIEWS})',
        f'reprojection rms: {rms:.3f} px   (good < 0.5, usable < 1.0)',
        f'per-view error: median {np.median(per):.3f}, max {np.max(per):.3f} px',
        f'image coverage: {cov * 100:.0f}% of an 8x6 grid   (want > 80%)',
        f'fx {K[0, 0]:.2f}  fy {K[1, 1]:.2f}  cx {K[0, 2]:.2f}  cy {K[1, 2]:.2f}  '
        f'(horizontal FOV {hfov:.0f} deg)',
        f'dist (k1 k2 p1 p2 k3): {np.round(D.ravel()[:5], 5).tolist()}',
    ]
    warn, bad = [], []
    if n_tilt < MIN_TILTED_VIEWS:
        bad.append(f'only {n_tilt} tilted views: focal length is not determined. '
                   f'Tilt the board 20-40 deg in both directions')
    if not (0.4 * size[0] < K[0, 0] < 1.8 * size[0]):
        bad.append(f'fx {K[0, 0]:.0f} is implausible for a {size[0]} px wide image')
    if rms > 1.0:
        warn.append('rms > 1 px: blurred captures or wrong sizes in the board yaml')
    if cov < 0.8:
        warn.append('board did not reach the image corners/edges: distortion poorly constrained')
    if abs(K[0, 2] - size[0] / 2) > 0.1 * size[0] or abs(K[1, 2] - size[1] / 2) > 0.1 * size[1]:
        warn.append('principal point far from the image centre: add tilted views')
    lines += ['WARNING: ' + w for w in warn]
    for ln in lines + ['REJECTED: ' + b for b in bad]:
        print(ln)
    if bad and not a.force:
        print('NOT saved (add more tilted views, or use --force)')
        return None
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'camera.yaml')
    save_ros_yaml(path, K, D, size, a.camera_name,
                  [f'charuco_intrinsics.py  {lines[0]}', f'{len(views)} views, rms {rms:.3f} px, '
                   f'coverage {cov * 100:.0f}%', lines[1]])
    with open(os.path.join(a.out, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'saved {path}')
    return K, D


# ----------------------------------------------------------------------------- modes
def from_images(a):
    folder = os.path.expanduser(a.images)
    files = sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg')))
    views, names, size = [], [], None
    for fn in files:
        gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = size or gray.shape[::-1]
        if gray.shape[::-1] != size:
            print(f'{os.path.basename(fn)}: different image size, skipped')
            continue
        d = a.board_obj.detect(gray)
        if d is None:
            print(f'{os.path.basename(fn)}: board not found')
            continue
        views.append((d[0], d[1]))
        names.append(os.path.basename(fn))
    print(f'{len(views)} / {len(files)} images with the board')
    if size is None:
        sys.exit(f'no images in {folder}')
    if a.f_guess is None:
        a.f_guess = 0.73 * size[0]
    report_and_save(a, views, size, names)


class RosSource:
    """Latest frame of a ROS 2 sensor_msgs/Image or CompressedImage topic (no cv_bridge)."""

    def __init__(self, topic):
        import threading
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CompressedImage, Image
        except ImportError:
            sys.exit('--topic needs ROS 2: run  source /opt/ros/humble/setup.bash  first')
        rclpy.init()
        self.rclpy, self.lock, self.frame = rclpy, threading.Lock(), None
        self.node = rclpy.create_node('charuco_intrinsics')
        typ = CompressedImage if topic.endswith('compressed') else Image
        self.node.create_subscription(typ, topic, self._cb, qos_profile_sensor_data)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()
        print(f'waiting for {topic} ({typ.__name__}) ...')

    def _cb(self, msg):
        if hasattr(msg, 'format'):                       # CompressedImage
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        else:
            enc = msg.encoding.lower()
            ch = {'mono8': 1, 'rgb8': 3, 'bgr8': 3}.get(enc)
            if ch is None:
                print(f'unsupported encoding {msg.encoding}')
                return
            img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
            img = img[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
            img = {'mono8': lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR),
                   'rgb8': lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR)}.get(enc, np.copy)(img)
        with self.lock:
            self.frame = img

    def read(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if self.frame is not None:
                    f, self.frame = self.frame, None     # each frame used once
                    return True, f
            time.sleep(0.005)
        return False, None

    def release(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def open_camera(a):
    dev = int(a.device) if str(a.device).isdigit() else a.device
    backend = cv2.CAP_V4L2 if sys.platform.startswith('linux') else cv2.CAP_ANY
    cap = cv2.VideoCapture(dev, backend)
    if a.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*a.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)                # focus changes the intrinsics
    if not cap.isOpened():
        sys.exit(f'cannot open camera {a.device}')
    return cap


def live(a):
    cap = RosSource(a.topic) if a.topic else open_camera(a)
    ok, frame = cap.read()
    if not ok:
        sys.exit('no frame (topic not published? ROS_DOMAIN_ID?)' if a.topic
                 else 'camera opened but no frame')
    size = frame.shape[1::-1]
    if not a.topic and size != (a.width, a.height):
        print(f'WARNING: asked {a.width}x{a.height}, camera gives {size[0]}x{size[1]}')
    if a.f_guess is None:
        a.f_guess = 0.73 * size[0]
    min_move = a.min_move_px if a.min_move_px else 0.0625 * size[0]
    img_dir = os.path.join(a.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    print(f'{a.topic or "camera " + str(a.device)}: {size[0]}x{size[1]}, images -> {img_dir}')
    print(__doc__.split('\n\n')[3])

    views, names, tilts = [], [], []
    auto, last_t, last_c, prev = False, 0.0, None, None
    result, undist = None, False
    while True:
        ok, frame = cap.read()
        if not ok:
            print('no frame')
            if cv2.waitKey(50) & 0xFF in (ord('q'), 27):
                break
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        d = a.board_obj.detect(gray)
        n = 0 if d is None else len(d[0])
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        motion = float(np.mean(cv2.absdiff(gray, prev))) if prev is not None else 99.0
        prev = gray
        tilt = tilt_deg(d[0], d[1], size, a.f_guess) if d is not None else 0.0
        n_tilted = sum(1 for t in tilts if t >= MIN_TILT_DEG)
        moved = None
        if d is not None and last_c is not None and len(last_c) == len(d[1]):
            moved = float(np.median(np.linalg.norm(last_c - d[1], axis=1)))

        if n < a.min_corners:
            reason = f'too few corners ({n}<{a.min_corners})'
        elif sharp < a.min_sharpness:
            reason = f'blurred ({sharp:.0f}<{a.min_sharpness:.0f})'
        elif motion >= a.still_threshold:
            reason = f'hold still ({motion:.1f})'
        elif time.monotonic() - last_t < 1.0:
            reason = 'cooldown'
        elif tilt < MIN_TILT_DEG and len(tilts) - n_tilted >= 4:
            reason = f'tilt the board! ({tilt:.0f} deg)'
        elif moved is not None and moved <= min_move:
            reason = f'move to a new spot ({moved:.0f}px)'
        else:
            reason = None

        vis = frame.copy()
        if undist and result is not None:
            vis = cv2.undistort(vis, *result)
        else:
            for _, img in views:
                for x, y in img[::2]:
                    cv2.circle(vis, (int(x), int(y)), 1, (255, 160, 0), -1)
            if d is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, d[2], d[3], (0, 255, 0))
        scale = max(1, int(round(640 / size[0])))        # small images: enlarge the window
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        h = vis.shape[0]
        txt = (f'views {len(views)}  corners {n}/{a.board_obj.n_corners}  '
               f'cov {coverage(views, size) * 100:.0f}%  tilt {tilt:.0f}  '
               f'tilted {n_tilted}/{MIN_TILTED_VIEWS}  auto {"ON" if auto else "off"}')
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if auto:
            cv2.putText(vis, 'AUTO: ' + (reason or 'capturing'), (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if reason is None else (0, 0, 255), 2)
        if undist:
            cv2.putText(vis, 'UNDISTORTED (v)', (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 0, 255), 2)
        cv2.putText(vis, 'SPACE cap  a auto  u undo  c calib  v undist  q quit', (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow('charuco_intrinsics', vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') or (auto and reason is None):
            if n < a.min_corners or sharp < a.min_sharpness:
                print(f'not captured: {reason}')
            else:
                name = f'{len(views):03d}.png'
                cv2.imwrite(os.path.join(img_dir, name), frame)
                views.append((d[0], d[1]))
                tilts.append(tilt)
                names.append(name)
                last_c, last_t = d[1].copy(), time.monotonic()
                print(f'captured {name}: {n} corners, tilt {tilt:.0f} deg')
        elif key == ord('a'):
            auto = not auto
        elif key == ord('u') and views:
            views.pop()
            tilts.pop()
            os.remove(os.path.join(img_dir, names.pop()))
            last_c = views[-1][1] if views else None
            print(f'removed last view, {len(views)} left')
        elif key == ord('c'):
            result = report_and_save(a, views, size, names) or result
        elif key == ord('v'):
            undist = not undist and result is not None
        elif key in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description='ChArUco camera intrinsic calibration (no ROS)')
    p.add_argument('--board', default=os.path.join(HERE, 'charuco_board.yaml'),
                   help='board yaml with MEASURED sizes (default: next to this script)')
    p.add_argument('--out', default='./calib_out')
    p.add_argument('--images', default='', help='calibrate from saved images instead of live')
    p.add_argument('--topic', default='', help='ROS 2 image topic (remote camera), e.g. /image_raw')
    p.add_argument('--device', default='0', help='camera index or /dev/videoN')
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--height', type=int, default=240)
    p.add_argument('--fps', type=float, default=15.0)
    p.add_argument('--fourcc', default='MJPG', help='MJPG / YUYV / "" (camera default)')
    p.add_argument('--camera-name', default='camera', help='camera_name written to the yaml')
    p.add_argument('--f-guess', type=float, default=None,
                   help='focal length start value [px] (default 0.73 * width, ~ 70 deg HFOV)')
    p.add_argument('--force', action='store_true', help='save even if the checks reject it')
    p.add_argument('--min-corners', type=int, default=12)
    p.add_argument('--min-sharpness', type=float, default=60.0, help='Laplacian variance')
    p.add_argument('--min-move-px', type=float, default=None, help='default 1/16 of width')
    p.add_argument('--still-threshold', type=float, default=2.0)
    a, _ = p.parse_known_args()                       # ignore --ros-args
    a.out = os.path.expanduser(a.out)
    a.board_obj = Board(os.path.expanduser(a.board))
    print(f'OpenCV {cv2.__version__}, board: {a.board_obj.name}')
    if a.images:
        from_images(a)
    else:
        live(a)


if __name__ == '__main__':
    main()
