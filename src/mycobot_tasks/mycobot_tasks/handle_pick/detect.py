"""Handle detection from the two ArUco markers: frames, tracking (camera moves to look
straight down), heights from the handle top, detection log / replay, height sweep."""
import math
import os
import time

import numpy as np

from mycobot_tasks.red_cube_hover import CAM, Stop, project
from mycobot_tasks.tray_handle import handle_from_markers


class DetectMixin:
    def frames(self, n, single=False):
        """n fresh frames (arm still) -> list of handle_from_markers results, each with
        px (smallest marker side seen [px]), off (midpoint distance from the image centre,
        fraction of the image width) and view (camera axis vs marker normal [deg]).
        single: one marker is enough (centre spacing measured at the detection, else the
        nominal --gap + --tile-size)."""
        a = self.args
        out, after, t0 = [], time.monotonic(), time.monotonic()
        got = 0
        while got < n and time.monotonic() - t0 < a.pin_timeout:
            img = self.node.fresh(after, timeout=1.0)
            if img is None:
                continue
            after = time.monotonic()
            got += 1
            T = self.node.T('base', CAM)
            if T is None:
                raise Stop(f'no TF base -> {CAM}')
            found = self.mp.detect(img)
            self.last = (after, found)
            r = handle_from_markers(found, self.id_l, self.id_r, T,
                                    (self.spacing or a.spacing or 0.054) if single else None)
            if r is None:
                continue
            mine = [found[i][0] for i in (self.id_l, self.id_r) if i in found]
            r['px'] = min(float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))
                          for c in mine)
            h, w = img.shape[:2]
            uv = project(r['p'], T, self.K, self.D)
            r['off'] = 1.0 if uv is None else math.hypot(uv[0] - w / 2, uv[1] - h / 2) / w
            r['view'] = math.degrees(math.acos(min(1.0, abs(float(T[:3, 2] @ r['normal'])))))
            out.append(r)
        return out

    def view_problem(self, res, two):
        """Why the view is not good enough for the final measurement (None = good)."""
        a = self.args
        need = max(1, a.frames // 2)
        if len(two) < need:
            sides = {r['side'] for r in res if r['n'] == 1}
            name = {self.id_l: 'L (ID %d)' % self.id_l, self.id_r: 'R (ID %d)' % self.id_r}
            return 'only marker ' + '/'.join(name[s] for s in sides if s in name) + ' seen'
        px = float(np.median([r['px'] for r in two]))
        if px < a.min_marker_px:
            return f'too far (markers {px:.0f} px < --min-marker-px {a.min_marker_px:.0f})'
        off = float(np.median([r['off'] for r in two]))
        if off > a.max_off_centre:
            return f'near the image edge ({off:.2f} of the width from the centre)'
        view = float(np.median([r['view'] for r in two]))
        if view > a.max_view:
            return f'seen at an angle ({view:.0f} deg > --max-view)'
        return None

    def detect_pin(self, region='tray'):
        """Detect; when only one marker is seen, the markers are small, off the image centre
        or seen at an angle, the camera moves to look straight down at the (estimated)
        handle and measures again, at most --recenter times. Nothing seen: the camera looks
        at --scan-step around the search point first."""
        a = self.args
        self.pin = None
        if self.holding:
            raise Stop('holding the tray: put it back first (b)')
        if a.go and self.node.js.get('gripper_controller', 1.0) < 0.5:
            self.say('opening the gripper first: open fingers stay out of the camera view')
            self.gripper(True)
            time.sleep(0.3)
        need = max(1, a.frames // 2)
        s = a.scan_step
        cx, cy = a.search_x, a.search_y
        scan = [(cx + dx, cy + dy) for dx, dy in ((s, 0), (0, -s), (-s, 0), (0, s))]
        looked, scanned, tried_last = None, False, not a.use_last
        for attempt in range(a.recenter + 1):
            q0 = self.node.q()
            res = self.frames(a.frames, single=True)
            q1 = self.node.q()
            if q0 is None or q1 is None or np.max(np.abs(q1 - q0)) > np.radians(0.5):
                raise Stop('the arm moved during detection: try again when it stands still')
            two = [r for r in res if r['n'] == 2]
            if len(res) < need:                       # nothing (usable) seen
                if a.track and not tried_last:
                    tried_last = True
                    last = self.last_tray(region)
                    if last is not None:
                        self.say(f'no tray {a.tray} markers here: looking at the last tray '
                                 f'position ({last[0] * 1000:.0f}, {last[1] * 1000:.0f}) mm first')
                        if self.look_last(last, region):
                            continue
                if a.track and not scanned and self.region(region) is not None:
                    scanned = True                    # the region set in RViz
                    self.scan_region(region, self.tray_seen)
                    continue
                if a.track and scan and attempt < a.recenter:
                    nxt = scan.pop(0)
                    self.say(f'no tray {a.tray} markers: scanning ({nxt[0] * 1000:.0f}, '
                             f'{nxt[1] * 1000:.0f}) mm')
                    self.track_to(nxt)
                    continue
                seen = sorted(self.last[1]) if self.last else []
                raise Stop(f'tray {a.tray} markers ID {self.id_l} + {self.id_r} not seen '
                           f'({len(res)} of {a.frames} frames, IDs seen: {seen})')
            why = self.view_problem(res, two)
            if why is None:
                return self.finish_detect(two)
            xy = np.median(np.array([r['p'][:2] for r in res]), axis=0)
            again = looked is not None and np.linalg.norm(xy - looked) < a.track_tol
            if not a.track or attempt == a.recenter or again:
                if len(two) >= need:                   # both seen: use it, say why not ideal
                    self.say(f'note: {why}')
                    return self.finish_detect(two)
                raise Stop(f'{why} after {attempt} camera moves')
            self.say(f'{why}: camera straight above the handle ({xy[0] * 1000:.0f}, '
                     f'{xy[1] * 1000:.0f}) mm ({attempt + 1}/{a.recenter})')
            self.track_to(xy)
            looked = xy
        raise Stop('detection failed')

    def track_to(self, xy):
        if not self.args.go:
            raise Stop('(dry run) the camera would move there first: run with --go')
        self.look_at(xy, limit_step=False)

    def finish_detect(self, two):
        a = self.args
        P = np.array([r['p'] for r in two])
        p = np.median(P, axis=0)
        yaws = np.array([r['yaw'] for r in two])
        yaw = math.atan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))
        spread = float(np.max(np.linalg.norm(P[:, :2] - p[:2], axis=1)))
        tilt = float(np.median([r['tilt'] for r in two]))
        spacing = float(np.median([r['spacing'] for r in two]))
        if spread > a.max_spread:
            raise Stop(f'handle position not stable ({spread * 1000:.1f} mm spread > '
                       f'--max-spread): the tray or the arm is moving?')
        if tilt > a.max_tilt:
            raise Stop(f'marker plane tilted {tilt:.0f} deg (> --max-tilt) seen from straight '
                       'above: the tray is not flat; grasping straight down refused')
        if a.spacing and abs(spacing - a.spacing) > a.max_spacing_err:
            raise Stop(f'marker centres {spacing * 1000:.1f} mm apart (tile-edge gap '
                       f'{(spacing - a.tile_size) * 1000:.1f} mm), expected '
                       f'{a.spacing * 1000:.1f} mm (--gap {(a.spacing - a.tile_size) * 1000:.1f}'
                       ' + --tile-size): wrong --gap / --marker-size, or markers moved?')
        self.spacing = spacing
        self.set_heights(p[2] - a.table_z)
        self.pin = dict(p=p, yaw=yaw)
        self._last_pin = p.copy()
        self.log_detection('tray', two)
        if a.save and not a.replay:
            self.save_detection(p, yaw, spacing, tilt)
        self.say(f'handle ({p[0] * 1000:.1f}, {p[1] * 1000:.1f}) mm, top '
                 f'{(p[2] - a.table_z) * 1000:.1f} mm over the surface (--table-z '
                 f'{a.table_z * 1000:.0f} mm), L->R yaw '
                 f'{math.degrees(yaw):.1f} deg, L-R {spacing * 1000:.1f} mm, tilt '
                 f'{tilt:.1f} deg, spread {spread * 1000:.1f} mm ({len(two)} frames, '
                 f'markers {np.median([r["px"] for r in two]):.0f} px)')
        return self.pin

    def set_heights(self, h):
        """Handle top h over the desk -> the desk-relative heights tray_pick uses."""
        a = self.args
        a.grasp_z = h - a.grasp_depth
        a.approach = h + a.approach_above
        a.servo_height = h + a.servo_above
        if a.tray_height and abs(h - a.tray_height) > a.tray_height_tol:
            self.say(f'WARNING: handle top {h * 1000:.0f} mm over --table-z, but --tray-height '
                     f'is {a.tray_height * 1000:.0f} mm: the surface is at '
                     f'{(a.table_z + h - a.tray_height) * 1000:.0f} mm if the tray stands flat '
                     '(fix --table-z), or the handle sticks out above the tray')
        if h < 0 or h > a.max_handle_h:
            raise Stop(f'handle top {h * 1000:.0f} mm over --table-z {a.table_z * 1000:.0f} mm: '
                       'the tray does not stand on that surface. Below the desk: pass '
                       '--table-z <z of the surface the tray stands on, base frame>')
        if a.grasp_z <= a.table_margin:
            raise Stop(f'grasp height {a.grasp_z * 1000:.0f} mm over the surface is inside '
                       f'--table-margin ({a.table_margin * 1000:.0f} mm): handle top only '
                       f'{h * 1000:.0f} mm high? reduce --grasp-depth')

    def measure_pin(self, n=None):
        """Servo: median midpoint xy (one marker + spacing allowed) or None."""
        n = n or self.args.frames
        res = self.frames(n, single=True)
        if len(res) <= n // 2:
            return None
        return np.median(np.array([r['p'][:2] for r in res]), axis=0)

    def save_detection(self, p, yaw, spacing, tilt):
        import yaml
        q = self.node.q()
        d = dict(stamp=time.strftime('%Y-%m-%d %H:%M:%S'), tray=self.args.tray,
                 handle_xyz=[float(v) for v in p], yaw_lr_deg=math.degrees(yaw),
                 spacing=spacing, tilt_deg=tilt, table_z=self.args.table_z,
                 camera_q_deg=[float(v) for v in np.degrees(q)] if q is not None else None)
        path = os.path.expanduser(self.args.save)
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(d, f, sort_keys=False)
            print(f'  detection saved: {path} (replay in simulation: --replay)', flush=True)
        except OSError as e:
            print(f'  detection not saved ({e})', flush=True)

    def load_replay(self):
        """--replay: the saved detection replaces the camera (frames())."""
        import yaml
        path = os.path.expanduser(self.args.replay)
        with open(path) as f:
            d = yaml.safe_load(f)
        p = np.array(d['handle_xyz'], float)
        r = dict(p=p, yaw=math.radians(d['yaw_lr_deg']), spacing=float(d['spacing']), n=2,
                 normal=np.array([0.0, 0.0, 1.0]), side=None, tilt=float(d['tilt_deg']),
                 px=100.0, off=0.0, view=float(d['tilt_deg']))
        self.frames = lambda n, single=False: [dict(r) for _ in range(n)]
        bot_path = os.path.expanduser(self.args.bot_save)
        if os.path.exists(bot_path):
            with open(bot_path) as f:
                b = yaml.safe_load(f)
            rb = dict(p=np.array(b['marker_xyz'], float), yaw=math.radians(b['yaw_deg']),
                      normal=np.array([0.0, 0.0, 1.0]), tilt=float(b['tilt_deg']), px=100.0,
                      off=0.0, view=0.0)
            self.bot_frames = lambda n: [dict(rb) for _ in range(n)]
            print(f"REPLAY TurtleBot marker {bot_path} ({b['stamp']})", flush=True)
        self.replay_q = None if d.get('camera_q_deg') is None else \
            np.radians(np.array(d['camera_q_deg'], float))
        print(f"REPLAY {path} ({d['stamp']}): handle ({p[0] * 1000:.1f}, {p[1] * 1000:.1f}, "
              f"{p[2] * 1000:.1f}) mm, L->R yaw {d['yaw_lr_deg']:.1f} deg; detected with "
              f"--table-z {d['table_z']}", flush=True)
        if abs(d['table_z'] - self.args.table_z) > 1e-4:
            print(f"  NOTE: --table-z {self.args.table_z} here, {d['table_z']} at the detection",
                  flush=True)

    def act_replay_start(self):
        """Replay: move the (simulated) arm to the pose the detection was made from."""
        if self.replay_q is None:
            return
        self.move('detection pose (replay)', self.replay_q)
        if self.args.go:
            self.home = self.replay_q.copy()
            time.sleep(0.3)
        self.say('at the detection pose of the replay: m = pick + place, p = pick')

    def log_detection(self, kind, res, note=''):
        """One CSV row per measurement (--detect-log): where the camera was and what it saw,
        to compare detections from different heights / angles / image positions."""
        a = self.args
        if not a.detect_log or not res:
            return None
        T = self.node.T('base', CAM)
        P = np.array([r['p'] for r in res])
        p = np.median(P, axis=0)
        cam = T[:3, 3] if T is not None else np.full(3, np.nan)
        q = self.node.q()
        row = dict(
            time=time.strftime('%Y-%m-%d %H:%M:%S'), kind=kind, note=note, frames=len(res),
            x_mm=p[0] * 1000, y_mm=p[1] * 1000, z_mm=p[2] * 1000,
            sx_mm=float(np.std(P[:, 0])) * 1000, sy_mm=float(np.std(P[:, 1])) * 1000,
            yaw_deg=math.degrees(math.atan2(np.median(np.sin([r['yaw'] for r in res])),
                                            np.median(np.cos([r['yaw'] for r in res])))),
            cam_x_mm=cam[0] * 1000, cam_y_mm=cam[1] * 1000, cam_z_mm=cam[2] * 1000,
            cam_h_mm=(cam[2] - p[2]) * 1000, dist_mm=float(np.linalg.norm(cam - p)) * 1000,
            view_deg=float(np.median([r['view'] for r in res])),
            off=float(np.median([r['off'] for r in res])),
            px=float(np.median([r['px'] for r in res])),
            tilt_deg=float(np.median([r['tilt'] for r in res])),
            q_deg=' '.join(f'{v:.2f}' for v in np.degrees(q)) if q is not None else '')
        path = os.path.expanduser(a.detect_log)
        try:
            new = not os.path.exists(path)
            with open(path, 'a') as f:
                if new:
                    f.write(','.join(row) + '\n')
                f.write(','.join(f'{v:.2f}' if isinstance(v, float) else str(v)
                                 for v in row.values()) + '\n')
        except OSError as e:
            print(f'  detection log not written ({e})', flush=True)
        return row

    def act_sweep(self):
        """g: the same target (tray handle, or the TurtleBot marker with --sweep-target bot)
        measured with the camera straight above it at each --sweep-heights (over the target),
        then at --sweep-offsets beside it (target off the image centre). Every measurement
        goes to --detect-log; the table compares them with the lowest straight view."""
        a = self.args
        if not a.go:
            raise Stop('(dry run) the sweep moves the camera: run with --go')
        bot = a.sweep_target == 'bot'

        def measure():
            if bot:
                res = self.bot_frames(a.frames)
            else:
                res = [r for r in self.frames(a.frames) if r['n'] == 2]
            return res if len(res) >= max(1, a.frames // 2) else None
        home = self.home
        res = measure()
        if res is None:
            if bot:
                self.find_bot()
            else:
                self.detect_pin()
            res = measure()
            if res is None:
                raise Stop('target not seen')
        p0 = np.median(np.array([r['p'] for r in res]), axis=0)
        x, y, z = (float(v) for v in p0)
        what = 'TurtleBot marker' if bot else 'tray handle'
        self.say(f'sweep: {what} at ({x * 1000:.0f}, {y * 1000:.0f}, {z * 1000:.0f}) mm, '
                 f'heights {a.sweep_heights}, offsets {a.sweep_offsets}')
        self.confirm(f'sweep the camera over the {what} ({len(a.sweep_heights)} heights)')
        rows = []

        def run(i, n, h, dx, dy):
            self._check()
            tag = f'h{h * 100:.0f}' + (f' off{dx * 100:+.0f},{dy * 100:+.0f}' if dx or dy else '')
            try:
                self.look_over(x + dx, y + dy, z, f'sweep {i}/{n} {tag}',
                               oblique=False, view=h, exact=True)
            except Stop as e:
                if 'no look-down pose' not in str(e):
                    raise
                print(f'  {tag}: not reachable, skipped', flush=True)
                return None
            res = measure()
            if res is None:
                print(f'  {tag}: target not seen', flush=True)
                return None
            row = self.log_detection(f'sweep_{a.sweep_target}', res, tag)
            if row:
                rows.append(row)
            return h
        n = len(a.sweep_heights)
        reached = [h for i, h in enumerate(a.sweep_heights)
                   if run(i + 1, n, h, 0.0, 0.0) is not None]
        if reached:                       # beside it: at the middle of the reached heights
            h_off = sorted(reached)[len(reached) // 2]
            offs = [(dx, dy) for d in a.sweep_offsets if d > 0
                    for dx, dy in ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d))]
            for i, (dx, dy) in enumerate(offs):
                run(i + 1, len(offs), h_off, dx, dy)
        rows = [r for r in rows if r]
        if not rows:
            raise Stop('sweep: nothing measured' + ('' if a.detect_log else
                                                    ' (--detect-log is off)'))
        straight = [r for r in rows if 'off' not in r['note']]
        ref = min(straight or rows, key=lambda r: r['cam_h_mm'])
        print(f'\nsweep result ({what}), reference = lowest straight view ({ref["note"]}):',
              flush=True)
        print('  view            cam_h   dist   view   off    px     x       y       z     '
              'dx    dy    dxy    dz   [mm, deg]', flush=True)
        for r in rows:
            dxr, dyr, dzr = (r['x_mm'] - ref['x_mm'], r['y_mm'] - ref['y_mm'],
                             r['z_mm'] - ref['z_mm'])
            print(f"  {r['note']:14s} {r['cam_h_mm']:6.0f} {r['dist_mm']:6.0f} "
                  f"{r['view_deg']:5.1f} {r['off']:5.2f} {r['px']:5.0f} {r['x_mm']:7.1f} "
                  f"{r['y_mm']:7.1f} {r['z_mm']:6.1f} {dxr:+5.1f} {dyr:+5.1f} "
                  f"{math.hypot(dxr, dyr):5.1f} {dzr:+5.1f}", flush=True)
        xy = np.array([[r['x_mm'], r['y_mm']] for r in rows])
        self.say(f'sweep done: {len(rows)} views, xy spread max '
                 f'{np.max(np.linalg.norm(xy - xy.mean(0), axis=1)):.1f} mm from the mean '
                 f'(log: {a.detect_log})')
        if home is not None:
            self.move('home', home)
