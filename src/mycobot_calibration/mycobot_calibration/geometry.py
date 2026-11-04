"""Small SE(3) helpers (numpy only)."""
import cv2
import numpy as np


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    """URDF fixed-axis roll-pitch-yaw (R = Rz(y) Ry(p) Rx(r))."""
    p = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(p)) > 1e-6:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: put everything in yaw
        r = 0.0
        y = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([r, p, y])


def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t).reshape(3)
    return T


def xyzrpy_to_T(v):
    return make_T(rpy_to_R(*v[3:6]), v[:3])


def T_to_xyzrpy(T):
    return np.concatenate([T[:3, 3], R_to_rpy(T[:3, :3])])


def rvec_tvec_to_T(rvec, tvec):
    return make_T(cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0], tvec)


def T_to_rvec_tvec(T):
    return cv2.Rodrigues(T[:3, :3])[0].reshape(3), T[:3, 3].copy()


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_angle(R):
    """Rotation angle [rad] of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def look_at(eye, target, up_hint):
    """Camera optical frame (z forward, x right, y down) at `eye` looking at `target`."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(-np.asarray(up_hint, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), eye)


def T_to_quat(T):
    """(x, y, z, w) quaternion of the rotation part."""
    R = T[:3, :3]
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform_msg_to_T(tr):
    """geometry_msgs/Transform -> 4x4."""
    q = tr.rotation
    return make_T(quat_to_R([q.x, q.y, q.z, q.w]),
                  [tr.translation.x, tr.translation.y, tr.translation.z])


def fill_pose_msg(pose, T):
    """Fill geometry_msgs/Pose from 4x4."""
    q = T_to_quat(T)
    pose.position.x, pose.position.y, pose.position.z = map(float, T[:3, 3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
    return pose


def fill_transform_msg(tr, T):
    q = T_to_quat(T)
    tr.translation.x, tr.translation.y, tr.translation.z = map(float, T[:3, 3])
    tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w = map(float, q)
    return tr
