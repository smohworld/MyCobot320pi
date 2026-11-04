"""Automatic calibration poses: cameras on a spherical cap looking at the board.

Each candidate camera pose is converted to a link6 target with the nominal (URDF)
camera extrinsics, solved with IK, and kept only if
  - IK converges inside the joint limits,
  - every link frame and the TCP stay above `min_z` (table) with clearance,
  - the joint-space path from the previous pose also stays above it,
  - the whole board is inside the image with a margin.
From the feasible set a diverse subset is picked (farthest-point in camera
orientation), then ordered to keep joint motions short.
"""
import numpy as np

from mycobot_calibration.geometry import inv, look_at, make_T, rot_angle


def _rot_about(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class PosePlanner:

    def __init__(self, chain, target, X_nominal, K, D, image_size, params):
        self.chain = chain
        self.target = target
        self.X = X_nominal
        self.K, self.D = K, D
        self.w, self.h = image_size
        self.p = params

    def camera_candidates(self, T_base_board, n, rng):
        p = self.p
        R_b = T_base_board[:3, :3]
        center = T_base_board[:3, :3] @ self.target.center + T_base_board[:3, 3]
        normal = -R_b[:, 2]                    # board z points away from the camera
        bx, by = R_b[:, 0], R_b[:, 1]
        out = []
        for _ in range(n):
            tilt = np.radians(rng.uniform(p['tilt_min_deg'], p['tilt_max_deg']))
            az = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(p['dist_min'], p['dist_max'])
            tilt_axis = np.cos(az) * bx + np.sin(az) * by
            view_dir = _rot_about(tilt_axis, tilt) @ normal
            aim = center + rng.uniform(-1, 1) * p['aim_jitter'] * bx \
                + rng.uniform(-1, 1) * p['aim_jitter'] * by
            eye = aim + d * view_dir
            T = look_at(eye, aim, up_hint=-by)
            roll = np.radians(rng.uniform(-p['roll_max_deg'], p['roll_max_deg']))
            T[:3, :3] = T[:3, :3] @ _rot_about([0, 0, 1], roll)
            out.append(T)
        return out

    def board_visible(self, T_base_cam, T_base_board):
        """Enough ChArUco corners inside the image (ChArUco works with a partial view)."""
        T_cb = inv(T_base_cam) @ T_base_board
        P = (T_cb[:3, :3] @ self.target.corners3d.T).T + T_cb[:3, 3]
        if np.any(P[:, 2] < 0.05):
            return False
        uv = (self.K @ (P / P[:, 2:3]).T).T[:, :2]
        m = self.p['image_margin']
        inside = ((uv[:, 0] > m * self.w) & (uv[:, 0] < (1 - m) * self.w)
                  & (uv[:, 1] > m * self.h) & (uv[:, 1] < (1 - m) * self.h))
        return bool(inside.mean() >= self.p['min_visible_frac'])

    def config_safe(self, q):
        frames = self.chain.fk(q, all_links=True)
        zmin = self.p['min_z'] + self.p['clearance']
        for name, T in frames.items():
            if name in self.p['ignore_links']:
                continue
            if T[2, 3] < zmin:
                return False
        # tool points rigidly attached to link6: camera lens, TCP / fingertips, ...
        T6 = frames[self.p['flange_link']]
        for T_rel in [self.X] + list(self.p.get('tool_points', [])):
            if (T6 @ T_rel)[2, 3] < zmin:
                return False
        return True

    def path_safe(self, q0, q1, steps=12):
        return all(self.config_safe(q0 + (q1 - q0) * s) for s in np.linspace(0, 1, steps))

    def plan(self, T_base_board, q_current, rng=None):
        rng = rng or np.random.default_rng(self.p['seed'])
        p = self.p
        feasible = []   # (T_base_cam, q)
        shoulder = np.array([0.0, 0.0, p['shoulder_z']])
        for T_cam in self.camera_candidates(T_base_board, p['n_candidates'], rng):
            if len(feasible) >= p['enough_feasible']:
                break
            if not self.board_visible(T_cam, T_base_board):
                continue
            T_l6 = T_cam @ inv(self.X)
            if np.linalg.norm(T_l6[:3, 3] - shoulder) > p['max_reach']:
                continue   # cheap reject before IK
            q = self.chain.ik(T_l6, q_current, restarts=2, iters=80, rng=rng)
            if q is None or not self.config_safe(q):
                continue
            feasible.append((T_cam, q))
        if not feasible:
            return [], 0

        # diversity: greedy farthest point on camera orientation (+ a bit of position)
        def dist(a, b):
            return (np.degrees(rot_angle(a[0][:3, :3].T @ b[0][:3, :3]))
                    + 100.0 * np.linalg.norm(a[0][:3, 3] - b[0][:3, 3]))

        chosen = [min(range(len(feasible)),
                      key=lambda i: np.abs(feasible[i][1] - q_current).sum())]
        dmin = np.array([dist(feasible[chosen[0]], f) for f in feasible])
        while len(chosen) < min(p['n_poses'], len(feasible)):
            i = int(np.argmax(dmin))
            if dmin[i] <= 0:
                break
            chosen.append(i)
            dmin = np.minimum(dmin, [dist(feasible[i], f) for f in feasible])

        # order: nearest neighbour in joint space, each hop must be path-safe
        remaining = [feasible[i] for i in chosen]
        ordered, q_prev = [], np.asarray(q_current, float)
        while remaining:
            remaining.sort(key=lambda f: np.abs(f[1] - q_prev).max())
            for k, f in enumerate(remaining):
                if self.path_safe(q_prev, f[1]):
                    ordered.append(f)
                    q_prev = f[1]
                    remaining.pop(k)
                    break
            else:
                break   # rest unreachable without a detour: drop them
        return ordered, len(feasible)


def default_params():
    return {
        'n_poses': 18,
        'n_candidates': 600,
        'enough_feasible': 80,
        'shoulder_z': 0.1739,      # joint2 height (myCobot 320)
        'max_reach': 0.42,         # shoulder -> link6 origin [m]
        'dist_min': 0.28,
        'dist_max': 0.42,
        'tilt_min_deg': 10.0,
        'tilt_max_deg': 35.0,
        'roll_max_deg': 35.0,
        'aim_jitter': 0.03,
        'image_margin': 0.04,
        'min_visible_frac': 0.75,
        'min_z': 0.0,
        'clearance': 0.04,
        'ignore_links': ['base', 'link1'],
        'flange_link': 'link6',
        'seed': 0,
    }


def nominal_board_pose(xyz, yaw):
    """Board lying face-up on the table (board z points down into the table)."""
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)    # flip: z down
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return make_T(Rz @ R, xyz)
