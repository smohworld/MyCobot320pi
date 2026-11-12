"""Viewpoint planning for intrinsics_auto: which camera poses fill the image grid.

No ROS node inside: the planner gets a parameter getter `g(name)`, a MoveItClient (IK only) and
a log callback, so the node (real run) and the board-position sweep share the same code.

  configure(board pose, hand-eye X, ...)   per board position / run; clears the pose cache
  plan(hit cells, tilted views, ...)       one round: re-use reachable poses from earlier rounds,
                                           top up with new pool poses (IK) up to max_reachable,
                                           then greedy pick (new image-grid cells, tilted views,
                                           variety)

Pose pool: camera on a spherical cap above the board (tilt, azimuth, distance, roll), board pushed
to the image border by moving the camera sideways (not by turning it: keeps the orientation
reachable). Filters before any IK: camera height window and distance from the shoulder axis
(reach_min..reach_max, measured on this arm: beyond ~0.30 m IK practically never succeeds).
Prediction: pinhole with a rough K, a chessboard corner counts only if a neighbouring ArUco marker
is fully inside the image (that is what the detector needs).
"""
import os

import cv2
import numpy as np

from mycobot_calibration.geometry import inv, look_at, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.intrinsics_calib import F_GUESS, MIN_TILT_DEG, MIN_TILTED_VIEWS
from mycobot_calibration.pose_generator import ViewpointGenerator, camera_roll

GRID = (8, 6)                              # same grid as intrinsics_calib.coverage()


def cells_of(pix, size, grid=GRID):
    """Set of (row, col) grid cells containing at least one of the pixels."""
    w, h = size
    gx = np.clip((pix[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip((pix[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
    return set(zip(gy.tolist(), gx.tolist()))


class ViewPlanner:

    def __init__(self, target, g, moveit, log=print, rng=None):
        self.target = target
        self.g = g
        self.moveit = moveit
        self.log = log                         # log(text, level='info')
        self.rng = rng if rng is not None else np.random.default_rng(int(g('seed')))
        self.K_prior = self.D_prior = None     # prediction intrinsics (None: pinhole f guess)
        self.ik_log_path = None                # csv with every IK trial (None: off)
        # chessboard corner -> adjacent ArUco markers: a corner is only detected when a marker
        # next to it is fully visible
        mk = np.array([m.reshape(4, 3) for m in target.board.getObjPoints()])
        self.marker_pts = mk
        ctr = mk.mean(1)[:, :2]
        dist = np.linalg.norm(target.corners3d[:, None, :2] - ctr[None], axis=2)
        self.adjacent = dist <= 0.75 * target.square          # (n_corners, n_markers)
        self.T_board = self.X = self.T_l6_ik = self.start_q = self.size = None
        self.T_cam_ref = None
        self.cache = []                        # reachable (T_cam, q, meta), not yet used
        self.meta = {}
        self.stats = {}

    # ------------------------------------------------------------------ setup
    def configure(self, T_board, X, T_l6_ik, start_q, size, T_cam_ref):
        """New board pose / start configuration: reachable poses found so far are invalid."""
        self.T_board, self.X, self.T_l6_ik = T_board, X, T_l6_ik
        self.start_q = np.asarray(start_q, float)
        self.size = size
        self.T_cam_ref = T_cam_ref             # camera pose that defines the roll reference
        self.cache = []

    def K_pred(self):
        """K used only to predict where the board lands (the real K is what we estimate)."""
        if self.K_prior is not None:
            return self.K_prior, self.D_prior
        w, h = self.size
        f = float(self.g('focal_px_guess')) if self.g('focal_px_guess') else float(F_GUESS)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), np.zeros(5)

    # ------------------------------------------------------------- prediction
    def predict(self, T_cam_board):
        """(cells hit, tilt deg, board-centre pixel) the board would give, or None."""
        g = self.g
        K, D = self.K_pred()
        w, h = self.size
        Pc = (T_cam_board[:3, :3] @ self.target.corners3d.T).T + T_cam_board[:3, 3]
        if np.any(Pc[:, 2] < 0.05):
            return None
        tilt = float(np.degrees(np.arccos(np.clip(abs(T_cam_board[2, 2]), 0, 1))))
        if tilt > g('max_incidence_deg'):
            return None
        rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
        tvec = T_cam_board[:3, 3]
        m = g('image_margin_px')

        def inside(pts3):
            px, _ = cv2.projectPoints(pts3, rvec, tvec, K, D)
            px = px.reshape(-1, 2)
            return px, ((px[:, 0] > m) & (px[:, 0] < w - m) & (px[:, 1] > m) & (px[:, 1] < h - m))

        _, min_ = inside(self.marker_pts.reshape(-1, 3))
        marker_ok = min_.reshape(-1, 4).all(axis=1)               # marker fully in the image
        px, cin = inside(self.target.corners3d)
        detected = cin & (self.adjacent & marker_ok[None]).any(axis=1)
        n_in = int(detected.sum())
        if n_in < max(g('min_corners'), g('min_visible_fraction') * self.target.n_corners):
            return None
        ctr = T_cam_board[:3, :3] @ self.target.center + T_cam_board[:3, 3]
        cpx = (K @ ctr)[:2] / ctr[2]
        return cells_of(px[detected], self.size), tilt, cpx

    # -------------------------------------------------------------- pose pool
    def sample_pool(self):
        """Camera poses (base frame) around the board that pass the cheap geometric filters."""
        g = self.g
        gen = ViewpointGenerator(self.target)
        Rb = self.T_board[:3, :3]
        c = Rb @ self.target.center + self.T_board[:3, 3]
        ref = gen.reference_frame(self.T_board, self.T_cam_ref[:3, 3])
        ref_roll = camera_roll(self.T_cam_ref, ref)
        nrm = -Rb[:, 2]
        e1 = Rb[:, 0]
        e2 = np.cross(nrm, e1)
        up = gen.up_hint(c)
        half = np.array(self.target.size) / 2
        shoulder = np.asarray(g('shoulder_xyz'), float)
        pool = []
        for _ in range(int(g('pool_size'))):
            tilt = np.radians(self.rng.uniform(g('tilt_min_deg'), g('tilt_max_deg')))
            az = self.rng.uniform(0, 2 * np.pi)
            d = self.rng.uniform(g('distance_min'), g('distance_max'))
            direction = np.cos(tilt) * nrm + np.sin(tilt) * (np.cos(az) * e1 + np.sin(az) * e2)
            # small aim jitter only: keep the orientation reachable (like the hand-eye poses)
            off = self.rng.uniform(-1, 1, 2) * g('aim_range') * half
            aim = c + off[0] * e1 + off[1] * e2
            T = look_at(c + d * direction, aim, up_hint=up)
            roll = ref_roll + np.radians(self.rng.uniform(-g('roll_range_deg'),
                                                          g('roll_range_deg')))
            T[:3, :3] = T[:3, :3] @ rot_z(roll)
            # push the board to the image border by MOVING the camera sideways (its own x/y):
            # shift / distance = image offset / focal length
            sh = self.rng.uniform(0.0, 1.0) * g('shift_range') * d
            ang = self.rng.uniform(0, 2 * np.pi)
            T[:3, 3] += sh * (np.cos(ang) * T[:3, 0] + 0.75 * np.sin(ang) * T[:3, 1])
            z = T[2, 3]
            if not g('table_z') + g('min_camera_height') <= z <= g('table_z') + g('max_camera_height'):
                continue
            reach = float(np.linalg.norm(T[:3, 3] - shoulder))
            if not g('reach_min') <= reach <= g('reach_max'):
                continue
            self.meta[id(T)] = dict(
                tilt=round(float(np.degrees(tilt)), 1), az=round(float(np.degrees(az)), 0),
                dist=round(float(d), 3), roll=round(float(np.degrees(roll - ref_roll)), 1),
                shift=round(float(sh / d), 2), cam_x=round(float(T[0, 3]), 3),
                cam_y=round(float(T[1, 3]), 3), cam_z=round(float(z), 3),
                reach=round(reach, 3))
            pool.append(T)
        return pool

    # --------------------------------------------------------------------- IK
    def _log_ik(self, T_cam, result, dev_deg=''):
        m = self.meta.get(id(T_cam))
        if m is None or self.ik_log_path is None:
            return
        new = not os.path.exists(self.ik_log_path)
        with open(self.ik_log_path, 'a') as f:
            if new:
                f.write('result,max_dev_deg,' + ','.join(m) + '\n')
            f.write(f'{result},{dev_deg},' + ','.join(str(v) for v in m.values()) + '\n')

    def ik(self, T_cam, dev):
        """Joint solution within the window around the start pose, or None. Failures are counted
        by reason: collision (solvable without the collision check), kinematic (limits /
        orientation), joint_window (max_joint_dev_deg)."""
        T_ik = T_cam @ inv(self.X) @ self.T_l6_ik
        link, timeout = self.g('ik_link'), float(self.g('ik_timeout'))
        q = self.moveit.compute_ik(T_ik, link, ARM_JOINTS, self.start_q, timeout=timeout)
        if q is None:
            q2 = self.moveit.compute_ik(T_ik, link, ARM_JOINTS, self.start_q, timeout=timeout,
                                        avoid_collisions=False)
            r = 'collision' if q2 is not None else 'kinematic'
            self.stats[r] += 1
            self._log_ik(T_cam, r)
            return None
        dd = np.abs(np.asarray(q) - self.start_q)
        if np.any(dd > dev):
            self.stats['joint_window'] += 1
            self._log_ik(T_cam, 'joint_window', round(float(np.degrees(dd.max())), 0))
            return None
        self.stats['ok'] += 1
        self._log_ik(T_cam, 'ok', round(float(np.degrees(dd.max())), 0))
        return np.array(q)

    # ------------------------------------------------------------------- plan
    def plan(self, hit, tilted, n_views, room, max_ik=None, abort=None):
        """One round. hit: grid cells already covered; tilted: tilted views so far; n_views: views
        so far; room: poses wanted at most.
        Returns (poses [(T_cam, q, tilt)], predicted covered cells, stats dict)."""
        g = self.g
        n_cells = GRID[0] * GRID[1]
        dev = np.radians(np.asarray(g('max_joint_dev_deg'), float))
        dev = np.where(dev > 0, dev, np.inf)
        max_ik = int(max_ik or g('max_ik'))
        want = int(g('max_reachable'))
        self.stats = {'ok': 0, 'collision': 0, 'kinematic': 0, 'joint_window': 0}

        # 1) poses found reachable in earlier rounds: no IK needed again (prediction is redone,
        #    the K used for it may have been refined)
        reach = []
        for T, q in self.cache:
            r = self.predict(inv(T) @ self.T_board)
            if r is not None:
                reach.append((T, q, *r))
        reused = len(reach)

        # 2) top up with new pool poses (shuffled: unbiased) until `want` are known reachable
        cand = []
        if reused < want:
            for T in self.sample_pool():
                r = self.predict(inv(T) @ self.T_board)
                if r is not None:
                    cand.append((T, *r))
            self.rng.shuffle(cand)
        ik_calls = 0
        for T, cells, tilt, cpx in cand:
            if ik_calls >= max_ik or len(reach) >= want or (abort is not None and abort.is_set()):
                break
            ik_calls += 1
            q = self.ik(T, dev)
            if q is not None:
                reach.append((T, q, cells, tilt, cpx))
                self.cache.append((T, q))
        self.log(f'planning: {reused} reachable poses re-used, {len(cand)} new candidates predicted '
                 f'visible, {ik_calls} IK calls {self.stats}')

        # 3) greedy pick among the reachable ones: most new cells, tilted views, variety
        covered = set(hit)
        chosen, picked = [], []
        while reach and len(chosen) < room:
            if (len(covered) / n_cells >= min(1.0, g('target_coverage') + g('plan_margin'))
                    and tilted >= MIN_TILTED_VIEWS
                    and n_views + len(chosen) >= g('min_views')):
                break
            scores = []
            for _, _, cells, tilt, cpx in reach:
                sc = float(len(cells - covered))
                if tilted < MIN_TILTED_VIEWS and tilt >= MIN_TILT_DEG + 3:
                    sc += 1.0                      # need tilted views
                nov = 1.0
                for _, _, _, t2, c2 in picked:
                    nov = min(nov, np.linalg.norm(cpx - c2) / 150.0 + abs(tilt - t2) / 25.0)
                scores.append(sc + 0.3 * nov)
            T, q, cells, tilt, cpx = reach.pop(int(np.argmax(scores)))
            chosen.append((T, q, tilt))
            picked.append((T, q, cells, tilt, cpx))
            covered |= cells
            if tilt >= MIN_TILT_DEG:
                tilted += 1
        used = {id(c[0]) for c in chosen}
        self.cache = [(T, q) for T, q in self.cache if id(T) not in used]   # captured: consumed
        stats = dict(self.stats, reused=reused, ik_calls=ik_calls, cache_left=len(self.cache))
        self.log(f'planned {len(chosen)} poses, predicted coverage '
                 f'{len(covered) / n_cells * 100:.0f}%, {len(self.cache)} reachable poses kept '
                 'for the next round')
        return chosen, covered, stats
