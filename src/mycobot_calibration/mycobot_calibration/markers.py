"""RViz MarkerArray helpers."""
import numpy as np

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from mycobot_calibration.geometry import fill_pose_msg

COLORS = {
    'planned': (0.6, 0.6, 0.6, 0.8),
    'current': (1.0, 0.85, 0.0, 1.0),
    'ok': (0.1, 0.85, 0.2, 1.0),
    'failed': (0.9, 0.1, 0.1, 1.0),
}


def _color(rgba):
    return ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))


def board_marker(target, T_base_board, frame, ns, alpha=1.0, tint=None, stamp=None):
    """Checkerboard squares as a CUBE_LIST (+ white backing), in the board frame."""
    back = Marker()
    back.header.frame_id = frame
    if stamp is not None:
        back.header.stamp = stamp
    back.ns, back.id, back.type, back.action = ns, 0, Marker.CUBE, Marker.ADD
    fill_pose_msg(back.pose, T_base_board)
    back.pose.position.x, back.pose.position.y, back.pose.position.z = map(
        float, T_base_board[:3, :3] @ np.array([*target.center[:2], 0.002]) + T_base_board[:3, 3])
    back.scale.x, back.scale.y, back.scale.z = target.size[0] + 0.02, target.size[1] + 0.02, 0.003
    back.color = _color(tint or (1, 1, 1, alpha))

    sq = Marker()
    sq.header = back.header
    sq.ns, sq.id, sq.type, sq.action = ns, 1, Marker.CUBE_LIST, Marker.ADD
    fill_pose_msg(sq.pose, T_base_board)
    s = target.square
    sq.scale.x, sq.scale.y, sq.scale.z = s, s, 0.001
    for i in range(target.cols):
        for j in range(target.rows):
            if (i + j) % 2 == 0:   # top-left square is black
                sq.points.append(Point(x=(i + 0.5) * s, y=(j + 0.5) * s, z=-0.0005))
                sq.colors.append(_color((0.05, 0.05, 0.05, alpha)))
    return [back, sq]


def camera_pose_markers(T_base_cam, frame, idx, state, ns='poses', size=0.05):
    """Small frustum + index label for one planned camera pose."""
    c = COLORS[state]
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id, m.type, m.action = ns, idx * 2, Marker.LINE_LIST, Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.003 if state != 'current' else 0.006
    m.color = _color(c)
    o = T_base_cam[:3, 3]
    a, b, d = size * 0.6, size * 0.35, size
    corners = [T_base_cam[:3, :3] @ np.array(v) + o
               for v in ([a, b, d], [-a, b, d], [-a, -b, d], [a, -b, d])]
    for k in range(4):
        m.points += [Point(x=o[0], y=o[1], z=o[2]),
                     Point(x=corners[k][0], y=corners[k][1], z=corners[k][2])]
        n = corners[(k + 1) % 4]
        m.points += [Point(x=corners[k][0], y=corners[k][1], z=corners[k][2]),
                     Point(x=n[0], y=n[1], z=n[2])]
    # "up" tick (image -y) so roll is visible
    up = T_base_cam[:3, :3] @ np.array([0, -b * 1.6, d]) + o
    top = (corners[2] + corners[3]) / 2
    m.points += [Point(x=top[0], y=top[1], z=top[2]), Point(x=up[0], y=up[1], z=up[2])]

    t = Marker()
    t.header.frame_id = frame
    t.ns, t.id, t.type, t.action = ns, idx * 2 + 1, Marker.TEXT_VIEW_FACING, Marker.ADD
    t.pose.position.x, t.pose.position.y, t.pose.position.z = map(float, o + [0, 0, 0.02])
    t.pose.orientation.w = 1.0
    t.scale.z = 0.02
    t.color = _color(c)
    t.text = str(idx)
    return [m, t]


def delete_all(frame, ns):
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.action = ns, Marker.DELETEALL
    return m
