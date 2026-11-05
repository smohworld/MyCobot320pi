"""Camera viewpoints for hand-eye calibration around a ChArUco board.

Each viewpoint: camera on a spherical cap above the board centre (tilt from the board
normal, azimuth around it, distance), looking at the centre plus a small offset, with a
roll about the optical axis. Rotation variety (tilt, azimuth and roll) is what makes
AX = XB well conditioned; translation alone does not help.
"""
import numpy as np

from mycobot_calibration.geometry import look_at, make_T, rot_angle, rot_z


def board_pose_flat(target, center_xyz, yaw):
    """Board lying face up on the table, centre at center_xyz, x axis rotated by yaw."""
    R = rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])       # z into the table
    return make_T(R, np.asarray(center_xyz, float) - R @ target.center)


def camera_roll(T_cam, T_ref):
    """Roll [rad] of T_cam about its optical axis relative to T_ref (same z assumed)."""
    Rr = T_ref[:3, :3].T @ T_cam[:3, :3]
    return float(np.arctan2(Rr[1, 0], Rr[0, 0]))


class ViewpointGenerator:

    def __init__(self, target, n_poses=15, dist=(0.20, 0.32), tilt_deg=(10.0, 35.0),
                 roll_range_deg=30.0, aim_jitter=0.25, seed=0):
        self.target = target
        self.n = int(n_poses)
        self.dist = dist
        self.tilt = np.radians(tilt_deg)
        self.roll_range = np.radians(roll_range_deg)
        self.aim_jitter = aim_jitter
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def up_hint(board_center):
        """Image 'up' reference: horizontal direction from the robot base to the board.

        Fixed in the world (not tied to the azimuth), so the roll the wrist needs stays
        in a small range and the joint6 window (camera cable) is respected.
        """
        u = np.array([board_center[0], board_center[1], 0.0])
        return u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-3 else np.array([1.0, 0, 0])

    def reference_frame(self, T_base_board, eye):
        c = T_base_board[:3, :3] @ self.target.center + T_base_board[:3, 3]
        return look_at(eye, c, up_hint=self.up_hint(c))

    def slots(self, T_base_board, ref_roll=0.0, max_slots=None):
        """Yield (slot index, list of candidate T_base_cam ordered by preference).

        Azimuths follow the golden angle, so any prefix of the sequence is spread evenly
        around the board. The caller keeps taking slots until it has n_poses reachable
        ones (some directions, e.g. far behind the board, are outside the arm's reach).
        """
        Rb = T_base_board[:3, :3]
        c = T_base_board[:3, :3] @ self.target.center + T_base_board[:3, 3]
        n = -Rb[:, 2]                         # board normal towards the camera
        e1 = Rb[:, 0]
        e2 = np.cross(n, e1)
        n_rings = 3 if self.n >= 9 else 2
        tilts = np.linspace(self.tilt[0], self.tilt[1], n_rings)
        rolls = [-self.roll_range, 0.0, self.roll_range]
        golden = np.pi * (3.0 - np.sqrt(5.0))   # 137.5 deg
        for i in range(max_slots or 3 * self.n):
            tilt = tilts[i % n_rings]
            az = golden * i + self.rng.uniform(-0.1, 0.1)
            d = self.rng.uniform(*self.dist)
            direction = np.cos(tilt) * n + np.sin(tilt) * (np.cos(az) * e1 + np.sin(az) * e2)
            jit = self.aim_jitter * np.array(self.target.size) * self.rng.uniform(-0.5, 0.5, 2)
            aim = c + jit[0] * e1 + jit[1] * e2
            pref = ref_roll + rolls[i % 3]
            cands = []
            # sampled distance first, then the near / far ends (the arm's reach is limited)
            for dd in dict.fromkeys((d, self.dist[0], self.dist[1])):
                base = look_at(c + dd * direction, aim, up_hint=self.up_hint(c))
                # preferred roll first, then further away from it
                for off in (0, 45, -45, 90, -90, 135, -135, 180):
                    T = base.copy()
                    T[:3, :3] = base[:3, :3] @ rot_z(pref + np.radians(off))
                    cands.append(T)
            yield i, cands


def select_diverse(feasible, n, q_start):
    """Farthest-point selection of n poses from (T_cam, q) pairs.

    Distance = relative camera rotation [deg] + 100 * camera position difference [m]
    (1 cm ~ 1 deg). Starts from the pose closest to q_start in joint space.
    """
    if len(feasible) <= n:
        return list(feasible)

    def dist(a, b):
        return (np.degrees(rot_angle(a[0][:3, :3].T @ b[0][:3, :3]))
                + 100.0 * np.linalg.norm(a[0][:3, 3] - b[0][:3, 3]))

    q0 = np.asarray(q_start, float)
    chosen = [int(np.argmin([np.abs(f[1] - q0).sum() for f in feasible]))]
    dmin = np.array([dist(feasible[chosen[0]], f) for f in feasible])
    while len(chosen) < n:
        i = int(np.argmax(dmin))
        if dmin[i] <= 0:
            break
        chosen.append(i)
        dmin = np.minimum(dmin, [dist(feasible[i], f) for f in feasible])
    return [feasible[i] for i in chosen]


def order_by_joint_distance(q_start, poses):
    """Greedy nearest-neighbour ordering (poses: list of (T_cam, q dict-like array))."""
    left = list(poses)
    out, q = [], np.asarray(q_start, float)
    while left:
        k = int(np.argmin([np.max(np.abs(np.asarray(p[1]) - q)) for p in left]))
        out.append(left.pop(k))
        q = np.asarray(out[-1][1])
    return out


def rotation_spread_deg(Ts):
    a = [np.degrees(rot_angle(Ts[i][:3, :3].T @ Ts[j][:3, :3]))
         for i in range(len(Ts)) for j in range(i + 1, len(Ts))]
    return (float(max(a)), float(np.median(a))) if a else (0.0, 0.0)
