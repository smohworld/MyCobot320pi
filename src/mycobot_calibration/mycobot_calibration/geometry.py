"""4x4 homogeneous transform helpers (numpy only)."""
import cv2
import numpy as np


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inv(T):
    # 회전행렬, 병진행렬
    # 회전행렬의 역은 회전행렬의 전치행렬이다.
    # 병진행렬의 역은 -R^T * t이다.
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_angle(R):
    """Rotation angle [rad] of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


# Angle-Axis (축-각) 회전행렬 만드는 함수
def rvec_tvec_to_T(rvec, tvec):
    return make_T(cv2.Rodrigues(np.asarray(rvec, float).reshape(3))[0], tvec)


def T_to_rvec_tvec(T):
    return cv2.Rodrigues(T[:3, :3])[0].reshape(3), T[:3, 3].copy()


def rpy_to_R(r, p, y):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    p = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(p)) > 1e-6:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:                                   # gimbal lock
        r = 0.0
        y = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([r, p, y])


def xyzrpy_to_T(v):
    return make_T(rpy_to_R(*v[3:6]), v[:3])


def T_to_xyzrpy(T):
    return np.concatenate([T[:3, 3], R_to_rpy(T[:3, :3])])


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """(x, y, z, w)"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, \
            (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, \
            (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, \
            (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, \
            (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def transform_msg_to_T(tf):
    """geometry_msgs/Transform -> 4x4"""
    q, t = tf.rotation, tf.translation
    return make_T(quat_to_R(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


def fill_transform_msg(tf, T):
    q = R_to_quat(T[:3, :3])
    tf.translation.x, tf.translation.y, tf.translation.z = (float(v) for v in T[:3, 3])
    tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = (float(v) for v in q)


def fill_pose_msg(pose, T):
    q = R_to_quat(T[:3, :3])
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = \
        (float(v) for v in q)


def look_at(eye, target, up_hint=(0.0, 0.0, 1.0)):
    """Camera optical frame (z forward, x right, y down) at eye looking at target."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up_hint, float)
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(-up, z)          # image x right, y down (y ~ -up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), eye)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
