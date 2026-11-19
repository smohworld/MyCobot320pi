"""Base app on red_cube_hover.App: step mode (confirm), camera look-down, fixed-yaw IK,
stdout log tee (formerly in tray_pick)."""
import os
import time

import numpy as np

from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import (App, CAM, DOWN, LIMITS, SEEDS_J23, Stop, TCP, look_at,
                                          ray_plane)


class BaseApp(App):
    def confirm(self, text):
        """Step mode: wait for Enter in the window (x / SPACE = Stop)."""
        if self.auto:
            return
        self._go_on.clear()
        self.prompt = text
        self.say(f'NEXT: {text}   [Enter = go, x = cancel, a = auto]')
        try:
            while not self._go_on.wait(0.1):
                self._check()
                if self.auto:
                    break
        finally:
            self.prompt = None

    def detect_cube(self):                # name kept: inherited actions (home, search) call it
        return self.detect_pin()

    def look_at(self, xy, limit_step=True):
        """Camera straight down toward desk point xy; it becomes home. With limit_step the step
        from the point the camera looks at now is limited to --look-step (a partly seen tray's
        blob centre is only an estimate); the point is kept within --look-max-r of the base."""
        a = self.args
        T_bc = self.node.T('base', CAM)
        if T_bc is None:
            raise Stop(f'no TF base -> {CAM}')
        now = ray_plane([self.K[0, 2], self.K[1, 2]], self.K, self.D, T_bc, a.table_z)[0][:2]
        d = np.asarray(xy, float) - now
        if limit_step and np.linalg.norm(d) > a.look_step:
            d *= a.look_step / np.linalg.norm(d)
        tgt = now + d
        r = np.linalg.norm(tgt)
        if r > a.look_max_r:
            tgt *= a.look_max_r / r
        self.say(f'finding a look-down pose over ({tgt[0] * 1000:.0f}, {tgt[1] * 1000:.0f}) mm')
        q, h = self.look_pose(tgt)
        self.target = np.array([tgt[0], tgt[1], a.table_z])
        self.confirm(f'camera {h * 100:.0f} cm above ({tgt[0] * 1000:.0f}, {tgt[1] * 1000:.0f}) mm')
        a.search_x, a.search_y = float(tgt[0]), float(tgt[1])
        self.move('look at tray', q)
        if a.go:
            self.home = q
            time.sleep(0.5)

    def look_pose(self, tgt):
        """Fast search_pose: 3 heights, 30 deg roll steps, few seeds, and each height stops after
        3 valid solutions (the full search can take minutes when the point is out of reach)."""
        a = self.args
        X = self.node.T('link6', CAM)
        T_l6_tcp = self.node.T('link6', TCP)
        q_now = self.node.q()
        if X is None or T_l6_tcp is None or q_now is None:
            raise Stop(f'no TF link6 -> {CAM} / {TCP} or joint_states')
        target = np.array([tgt[0], tgt[1], a.table_z])
        j1 = float(np.arctan2(tgt[1], tgt[0]))
        seeds = [q_now] + [np.array([j1, j2, j3, 0.0, 0.0, 0.0]) for j2, j3 in SEEDS_J23]
        min_tcp_z = a.table_z + a.table_margin + 0.02
        t0, calls = time.monotonic(), 0
        for h in (a.search_height, (a.search_height + a.search_min_height) / 2,
                  a.search_min_height):
            T0 = look_at(target + [0.0, 0.0, h], target)
            best, found = None, 0
            for roll in np.radians(np.arange(0, 360, 30)):
                if found >= 3:
                    break
                T_ik = T0 @ make_T(rot_z(roll), np.zeros(3)) @ np.linalg.inv(X) @ T_l6_tcp
                if T_ik[2, 3] < min_tcp_z:
                    continue
                for sd in seeds:
                    self._check()
                    calls += 1
                    q = self.mv.compute_ik(T_ik, TCP, ARM_JOINTS, sd, timeout=0.05)
                    if q is None:
                        continue
                    q = np.asarray(q)
                    if np.min(1 - np.abs(q) / LIMITS) <= 0.05:
                        continue
                    cost = 10.0 * (q[0] - j1) ** 2 + float(np.sum((q - q_now) ** 2))
                    found += 1
                    if best is None or cost < best[0]:
                        best = (cost, q)
                    break                            # this roll is solved: next roll
            if best is not None:
                print(f'  look pose: {calls} IK calls, {time.monotonic() - t0:.1f} s', flush=True)
                return best[1], h
        raise Stop(f'no look-down pose over ({tgt[0] * 1000:.0f}, {tgt[1] * 1000:.0f}) mm: '
                   'move the tray closer to the robot')

    def wait_still(self, timeout=0.6, tol_deg=0.03):
        """Until the measured joints stop changing (instead of a fixed settle sleep)."""
        t0, q0 = time.monotonic(), self.node.q()
        while time.monotonic() - t0 < timeout:
            time.sleep(0.06)
            q1 = self.node.q()
            if q0 is not None and q1 is not None and \
                    np.max(np.abs(q1 - q0)) < np.radians(tol_deg):
                return
            q0 = q1

    def ik_fixed(self, p, yaw, seed):
        """TCP at p pointing down with exactly this yaw (the held tray must not turn).
        The seed first; near the edge of the workspace (e.g. low below the desk) the solutions
        are a narrow band, so other seeds are tried too and the one closest to the seed wins."""
        T = make_T(rot_z(yaw) @ DOWN, p)
        seed = np.asarray(seed, float)
        q = self.mv.compute_ik(T, TCP, ARM_JOINTS, seed, timeout=0.1)
        if q is not None:
            return np.asarray(q)
        j1 = float(np.arctan2(p[1], p[0]))
        seeds = [np.array([j1, j2, j3, 0.0, 0.0, 0.0]) for j2, j3 in SEEDS_J23] + [
            np.array([j1, a2, a3, a4, 0.0, 0.0]) for a2, a3, a4 in
            ((-0.8, -1.2, 0.4), (0.3, -1.8, 0.0), (-1.2, -0.6, 0.3), (0.6, -2.0, -0.2))]
        best = None
        for sd in seeds:
            self._check()
            q = self.mv.compute_ik(T, TCP, ARM_JOINTS, sd, timeout=0.1)
            if q is not None:
                cost = float(np.max(np.abs(np.asarray(q) - seed)))
                if best is None or cost < best[0]:
                    best = (cost, np.asarray(q))
        return None if best is None else best[1]

    def ik_why(self, p, yaw, seed):
        """Why ik_fixed failed: 'collision' (solvable without the collision check) or 'reach'."""
        T = make_T(rot_z(yaw) @ DOWN, p)
        for sd in [seed] + [np.array([0.0, j2, j3, 0.0, 0.0, 0.0]) for j2, j3 in SEEDS_J23]:
            if self.mv.compute_ik(T, TCP, ARM_JOINTS, sd, timeout=0.1,
                                  avoid_collisions=False) is not None:
                return 'collision (desk / scene objects)'
        return 'out of reach / joint limits'


class _Tee:
    """stdout -> terminal + log file (timestamped lines), so long messages can be read later."""

    def __init__(self, stream, path):
        self.stream, self.f, self.bol = stream, open(path, 'a', buffering=1), True

    def write(self, s):
        self.stream.write(s)
        for part in s.splitlines(keepends=True):
            if self.bol:
                self.f.write(time.strftime('%H:%M:%S '))
            self.f.write(part)
            self.bol = part.endswith('\n')
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, n):
        return getattr(self.stream, n)


def _tee_stdout(path):
    import sys
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sys.stdout = _Tee(sys.stdout, path)
    print(f'--- tray_handle_pick {time.strftime("%Y-%m-%d %H:%M:%S")} '
          f'{" ".join(sys.argv[1:])}  (log: {path})', flush=True)
