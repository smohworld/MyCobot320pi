"""Tray / TurtleBot region scan: look-down views over a region (RViz boxes in
~/.ros/regions.yaml), chase a tray seen at the image edge, last known positions."""
import math
import os
import time

import cv2
import numpy as np

from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import CAM, LIMITS, SEEDS_J23, Stop, TCP, look_at, ray_plane
from mycobot_tasks import regions as rg


class ScanMixin:
    def region(self, name):
        """Region from regions.yaml (read now: edits in RViz count) or None."""
        return rg.saved(name, os.path.expanduser(self.args.regions))

    def view_height(self, r):
        """Camera height over the region surface: --scan-height when given, else the
        region's view (regions.yaml)."""
        a = self.args
        return a.scan_height if a.scan_height is not None else float(r.get('view', 0.28))

    def region_views(self, r):
        """Camera look points covering the region (snake order), nearest corner first.
        Footprint at the view height over the surface, overlapped by --scan-overlap."""
        a = self.args
        H = self.view_height(r)
        w, h = 2 * self.K[0, 2], 2 * self.K[1, 2]
        foot = min(w / self.K[0, 0], h / self.K[1, 1]) * H * (1.0 - a.scan_overlap)
        (x0, x1), (y0, y1) = r['x'], r['y']
        nx = max(1, math.ceil((x1 - x0) / foot))
        ny = max(1, math.ceil((y1 - y0) / foot))
        xs = [x0 + (i + 0.5) * (x1 - x0) / nx for i in range(nx)]
        ys = [y0 + (j + 0.5) * (y1 - y0) / ny for j in range(ny)]
        pts = [(x, y) for i, x in enumerate(xs) for y in (ys if i % 2 == 0 else ys[::-1])]
        T_bc = self.node.T('base', CAM)
        if T_bc is not None and len(pts) > 1:      # start at the end nearer the camera
            c = T_bc[:2, 3]
            if np.linalg.norm(np.array(pts[-1]) - c) < np.linalg.norm(np.array(pts[0]) - c):
                pts = pts[::-1]
        return pts, foot

    def look_over(self, x, y, z, name='scan view', oblique=True, view=None, exact=False):
        """Camera at (x, y) on the surface z: straight down from --scan-height (or lower,
        down to --search-min-height); out of reach and oblique: from up to --scan-pull
        closer to the base, still aimed at the point (search only, the measurement
        recentres). Does not change home. -> camera height."""
        a = self.args
        X = self.node.T('link6', CAM)
        T_l6_tcp = self.node.T('link6', TCP)
        q_now = self.node.q()
        if X is None or T_l6_tcp is None or q_now is None:
            raise Stop(f'no TF link6 -> {CAM} / {TCP} or joint_states')
        target = np.array([x, y, z])
        r = math.hypot(x, y)
        u = np.array([x / r, y / r, 0.0]) if r > 1e-6 else np.zeros(3)
        H = view if view is not None else (a.scan_height or 0.28)
        hs = [H] if exact else sorted(
            {H, (H + a.search_min_height) / 2, min(a.search_min_height, H)}, reverse=True)
        pulls = [0.0] + ([d for d in (0.06, 0.12, 0.18, 0.24) if d <= a.scan_pull + 1e-9]
                         if oblique else [])
        j1 = float(np.arctan2(y, x))
        seeds = [q_now] + [np.array([j1, j2, j3, 0.0, 0.0, 0.0]) for j2, j3 in SEEDS_J23]
        Xi = np.linalg.inv(X)
        min_tip = -np.inf
        if self.holding:          # the held tray's bottom stays over the rim + --carry-clear
            min_tip = z + a.bot_rim + a.carry_clear + a.tray_height - a.grasp_depth
        for pull in pulls:
            for h in hs:
                eye = target + [0.0, 0.0, h] - u * pull
                T0 = look_at(eye, target)
                best, found = None, 0
                for roll in np.radians(np.arange(0, 360, 30)):
                    if found >= 3:
                        break
                    T_ik = T0 @ make_T(rot_z(roll), np.zeros(3)) @ Xi @ T_l6_tcp
                    if T_ik[2, 3] < min_tip:
                        continue
                    for sd in seeds:
                        self._check()
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
                        break
                if best is not None:
                    if pull:
                        print(f'  {name}: oblique view, camera {pull * 100:.0f} cm closer to '
                              f'the base ({math.degrees(math.atan2(pull, h)):.0f} deg)',
                              flush=True)
                    self.move(name, best[1])
                    self.wait_still(timeout=1.0)
                    time.sleep(0.2)
                    return h
        raise Stop(f'no look-down pose over ({x * 1000:.0f}, {y * 1000:.0f}) mm'
                   + (' with the held tray over the rim (raise the camera height)'
                      if self.holding else ''))

    def scan_region(self, name, seen):
        """Look down at every view point of the region until seen() is True.
        -> (x, y) of that view. Unreachable view points are skipped."""
        a = self.args
        r = self.region(name)
        if r is None:
            raise Stop(f'no {rg.LABELS[name]} in {a.regions}: run '
                       '`ros2 run mycobot_tasks regions` and set it in RViz')
        if name == 'tray' and self.table_from_region:
            a.table_z = r['z']
        if not a.go:
            raise Stop(f'(dry run) scanning the {rg.LABELS[name]} needs camera moves: '
                       'run with --go')
        pts, foot = self.region_views(r)
        self.say(f'scanning the {rg.LABELS[name]}: {len(pts)} views, camera '
                 f'{self.view_height(r) * 100:.0f} cm over z {r["z"] * 1000:.0f} mm, '
                 f'({foot * 100:.0f} cm apart) | {rg.describe(name, r)}')
        self.confirm(f'scan the {rg.LABELS[name]} ({len(pts)} camera views)')
        skipped = 0
        for i, (x, y) in enumerate(pts):
            self._check()
            try:
                self.look_over(x, y, r['z'], f'{name} scan {i + 1}/{len(pts)}',
                               view=self.view_height(r))
            except Stop as e:
                if 'no look-down pose' not in str(e) and 'motion failed' not in str(e):
                    raise
                skipped += 1
                print(f'  view ({x * 1000:.0f}, {y * 1000:.0f}) mm not reachable, skipped',
                      flush=True)
                continue
            if seen():
                self.say(f'{rg.LABELS[name]}: found at view {i + 1}/{len(pts)} '
                         f'({x * 1000:.0f}, {y * 1000:.0f}) mm')
                return x, y
            got = self.chase_edge(name, r, seen)
            if got is not None:
                return got
        raise Stop(f'nothing found in the {rg.LABELS[name]} ({len(pts)} views'
                   + (f', {skipped} not reachable' if skipped else '') + ')')

    def edge_candidate(self, name, img, T_bc, z):
        """A dark patch cut by the image border that could be a marker partly outside the
        image (ArUco cannot read a cut marker) -> (x, y) on the surface z where the camera
        should look, or None. Size check against the marker (tray: also the tray body)."""
        a = self.args
        size = a.marker_size if name == 'tray' else a.bot_marker_size
        h_cam = max(0.05, float(T_bc[2, 3] - z))
        s = size * self.K[0, 0] / h_cam                       # expected marker side [px]
        big = (0.20 / size) ** 2 if name == 'tray' else 1.6  # tray body up to ~20 cm
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hh, ww = gray.shape
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = ((gray <= min(max(otsu, 30.0), a.chase_dark)) * 255).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cs:
            x, y, bw, bh = cv2.boundingRect(c)
            edge = []
            if x <= 2:
                edge.append((-1, 0))
            if y <= 2:
                edge.append((0, -1))
            if x + bw >= ww - 2:
                edge.append((1, 0))
            if y + bh >= hh - 2:
                edge.append((0, 1))
            if not edge or len(edge) > 2:          # not cut, or a huge dark area
                continue
            area = cv2.contourArea(c)
            if area < 0.15 * s * s or area > big * s * s or area < 0.6 * bw * bh:
                continue
            if best is None or area > best[0]:
                m = cv2.moments(c)
                best = (area, m['m10'] / m['m00'], m['m01'] / m['m00'], edge)
        if best is None:
            return None
        _, u, v, edge = best
        du = sum(e[0] for e in edge) * s / 2                  # marker centre is beyond
        dv = sum(e[1] for e in edge) * s / 2
        p = ray_plane([u + du, v + dv], self.K, self.D, T_bc, z)[0]
        return float(p[0]), float(p[1]), (u, v)

    def chase_edge(self, name, r, seen):
        """After a scan view without a marker: follow a cut patch at the image border
        (edge_candidate) up to --chase times. -> (x, y) when seen() there, else None."""
        a = self.args
        for k in range(a.chase):
            self._check()
            img = self.node.fresh(time.monotonic() - 0.05, timeout=1.0)
            T_bc = self.node.T('base', CAM)
            if img is None or T_bc is None:
                return None
            c = self.edge_candidate(name, img, T_bc, r['z'])
            if c is None:
                return None
            x, y, (u, v) = c
            self.say(f'{rg.LABELS[name]}: something cut by the image edge at px '
                     f'({u:.0f}, {v:.0f}): camera over ({x * 1000:.0f}, {y * 1000:.0f}) mm '
                     f'({k + 1}/{a.chase})')
            try:
                self.look_over(x, y, r['z'], f'{name} chase {k + 1}', view=self.view_height(r))
            except Stop as e:
                if 'no look-down pose' not in str(e) and 'motion failed' not in str(e):
                    raise
                print(f'  chase: {e}', flush=True)
                return None
            if seen():
                self.say(f'{rg.LABELS[name]}: found after following the image edge '
                         f'({x * 1000:.0f}, {y * 1000:.0f}) mm')
                return x, y
        return None

    def last_tray(self, region):
        """Where to look first for the tray: the last detection of this run, else the
        saved one (--save), as (x, y, surface z) -- only when it lies in the region (when
        that is set; a margin of 5 cm) or no region is set."""
        a = self.args
        p = None
        if getattr(self, '_last_pin', None) is not None:
            p = self._last_pin
        elif a.save:
            import yaml
            try:
                with open(os.path.expanduser(a.save)) as f:
                    d = yaml.safe_load(f)
                if int(d.get('tray', a.tray)) == a.tray:
                    p = np.array(d['handle_xyz'], float)
            except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
                p = None
        if p is None:
            return None
        r = self.region(region)
        if r is not None and not (r['x'][0] - 0.05 <= p[0] <= r['x'][1] + 0.05
                                  and r['y'][0] - 0.05 <= p[1] <= r['y'][1] + 0.05):
            return None
        return float(p[0]), float(p[1]), float(p[2]) - a.tray_height

    def saved_bot_xyz(self):
        if self.bot is not None:
            return tuple(float(v) for v in self.bot['p'])
        import yaml
        try:
            with open(os.path.expanduser(self.args.bot_save)) as f:
                d = yaml.safe_load(f)
            return tuple(float(v) for v in d['marker_xyz'])
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            return None

    def look_last(self, xyz, region):
        """Camera over a remembered position (view height of the region). -> moved?"""
        if not self.args.go:
            return False
        r = self.region(region)
        try:
            self.look_over(xyz[0], xyz[1], xyz[2], 'last position',
                           view=self.view_height(r) if r else None)
            return True
        except Stop as e:
            if 'no look-down pose' not in str(e) and 'motion failed' not in str(e):
                raise
            print(f'  last position: {e}', flush=True)
            return False

    def tray_seen(self):
        return len(self.frames(2, single=True)) > 0

    def act_find_tray(self):
        """f: scan the tray region from the start, then detect as d."""
        self.scan_region('tray', self.tray_seen)
        self.detect_pin()
