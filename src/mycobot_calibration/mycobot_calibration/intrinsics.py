"""Load / save camera intrinsics (K, D, image size).

Accepted yaml formats:
  * OpenCV FileStorage (cv2.FileStorage .yaml/.xml): camera_matrix, distortion_coefficients
    (or dist_coeffs), image_width, image_height
  * ROS camera_calibration / camera_info_manager: camera_matrix: {data: [...]},
    distortion_coefficients: {data: [...]}, image_width, image_height
  * plain yaml: K: [9 or 3x3], D: [...], width, height
"""
import os

import cv2
import numpy as np
import yaml


def _from_plain(c):
    def get(*keys):
        for k in keys:
            if k in c and c[k] is not None:
                return c[k]
        return None

    K = get('camera_matrix', 'K', 'k')
    D = get('distortion_coefficients', 'dist_coeffs', 'D', 'd')
    if isinstance(K, dict):
        K = K.get('data')
    if isinstance(D, dict):
        D = D.get('data')
    if K is None:
        raise ValueError('no camera_matrix / K')
    K = np.array(K, float).reshape(3, 3)
    D = np.array(D if D is not None else [0, 0, 0, 0, 0], float).reshape(-1)
    w = get('image_width', 'width')
    h = get('image_height', 'height')
    return K, D, (int(w), int(h)) if w and h else None


def load_intrinsics(path):
    path = os.path.expanduser(path)
    with open(path) as f:
        text = f.read()
    if text.lstrip().startswith('%YAML') or '!!opencv-matrix' in text or path.endswith('.xml'):
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode('camera_matrix').mat()
        D = fs.getNode('distortion_coefficients').mat()
        if D is None:
            D = fs.getNode('dist_coeffs').mat()
        w, h = fs.getNode('image_width').real(), fs.getNode('image_height').real()
        fs.release()
        if K is None:
            raise ValueError(f'{path}: no camera_matrix')
        D = np.zeros(5) if D is None else D.reshape(-1)
        return K.astype(float), D.astype(float), (int(w), int(h)) if w and h else None
    return _from_plain(yaml.safe_load(text))


def save_intrinsics(path, K, D, size, rms=None):
    """ROS camera_info_manager format (also readable by load_intrinsics)."""
    w, h = size
    data = {
        'image_width': int(w), 'image_height': int(h), 'camera_name': 'camera',
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in np.ravel(K)]},
        'distortion_model': 'plumb_bob',
        'distortion_coefficients': {'rows': 1, 'cols': len(np.ravel(D)),
                                    'data': [float(v) for v in np.ravel(D)]},
        'rectification_matrix': {'rows': 3, 'cols': 3,
                                 'data': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        'projection_matrix': {'rows': 3, 'cols': 4, 'data': [
            float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
            0.0, float(K[1, 1]), float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]},
    }
    with open(os.path.expanduser(path), 'w') as f:
        if rms is not None:
            f.write(f'# reprojection rms {rms:.3f} px\n')
        yaml.safe_dump(data, f, sort_keys=False)


def camera_info_msg(K, D, size, frame_id):
    from sensor_msgs.msg import CameraInfo
    m = CameraInfo()
    m.header.frame_id = frame_id
    m.width, m.height = int(size[0]), int(size[1])
    m.distortion_model = 'plumb_bob'
    m.k = [float(v) for v in np.ravel(K)]
    m.d = [float(v) for v in np.ravel(D)]
    m.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    m.p = [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
           0.0, float(K[1, 1]), float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
    return m
