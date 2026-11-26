"""TurtleBot: marker detection -> place spot / height / tray direction, its planning
scene objects, and r (the tray from the TurtleBot back to where m picked it)."""
import contextlib
import math
import os
import time

import cv2
import numpy as np

from mycobot_tasks.red_cube_hover import CAM, TCP, Stop, project


class BotMixin:
    def bot_frames(self, n, mid=None, size=None, timeout=None):
        """n fresh frames -> TurtleBot marker (--bot-id, --bot-marker-size) results:
        dict(p centre xyz base, yaw of the marker +x, normal, tilt, px, off, view).
        mid / size: another marker of the same dictionary (e.g. the heading marker): the
        pose found with --bot-marker-size is scaled to its side (tvec ~ marker size)."""
        a = self.args
        mid = a.bot_id if mid is None else mid
        k = 1.0 if size is None else size / a.bot_marker_size
        if self._settle_now in ('once', 'fast'):
            self.wait_still()      # the last (--mid-settle) motion may end while moving
        out, after, t0, got = [], time.monotonic(), time.monotonic(), 0
        while got < n and time.monotonic() - t0 < (a.pin_timeout if timeout is None
                                                   else timeout):
            img = self.node.fresh(after, timeout=1.0)
            if img is None:
                continue
            after = time.monotonic()
            got += 1
            T = self.node.T('base', CAM)
            if T is None:
                raise Stop(f'no TF base -> {CAM}')
            found = self.bot_mp.detect(img)
            if mid not in found:
                continue
            c, rv, tv, _ = found[mid]
            R = T[:3, :3] @ cv2.Rodrigues(rv)[0]
            p = T[:3, :3] @ (k * tv.reshape(3)) + T[:3, 3]
            nrm = R[:, 2] if R[2, 2] > 0 else -R[:, 2]
            hgt, wid = img.shape[:2]
            uv = project(p, T, self.K, self.D)
            out.append(dict(
                p=p, yaw=math.atan2(R[1, 0], R[0, 0]), normal=nrm,
                tilt=math.degrees(math.acos(min(1.0, abs(R[2, 2])))),
                px=float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1))),
                off=1.0 if uv is None else math.hypot(uv[0] - wid / 2, uv[1] - hgt / 2) / wid,
                view=math.degrees(math.acos(min(1.0, abs(float(T[:3, 2] @ nrm)))))))
        return out

    def find_bot(self):
        """find_bot_ with the TurtleBot area kept out (bot_search): its pose is not known
        until the marker is measured."""
        with self.bot_search():
            return self.find_bot_()

    def find_bot_(self):
        """Find the TurtleBot marker (here, else scan the TurtleBot region), look straight
        down at it, measure -> self.bot with the place spot / height / tray direction, and
        (--bot-scene) the TurtleBot + box into the planning scene at that pose."""
        a = self.args
        need = max(1, a.frames // 2)
        res = self.bot_frames(3)
        if not res and a.use_last:
            last = self.saved_bot_xyz()
            if last is not None:
                self.say(f'no TurtleBot marker here: looking at its last position '
                         f'({last[0] * 1000:.0f}, {last[1] * 1000:.0f}) mm first')
                if self.look_last(last, 'bot'):
                    res = self.bot_frames(3)
        if not res:
            self.scan_region('bot', lambda: len(self.bot_frames(2)) > 0)
        for attempt in range(3):
            res = self.bot_frames(a.frames)
            if len(res) < need:
                raise Stop(f'TurtleBot marker ID {a.bot_id} lost ({len(res)} of {a.frames} '
                           'frames)')
            p = np.median(np.array([r['p'] for r in res]), axis=0)
            off = float(np.median([r['off'] for r in res]))
            view = float(np.median([r['view'] for r in res]))
            if (off <= a.max_off_centre and view <= a.max_view) or not a.go or attempt == 2:
                break
            self.say(f'TurtleBot marker off the centre / at an angle (off {off:.2f}, view '
                     f'{view:.0f} deg): camera straight above it')
            try:
                rb = self.region('bot')
                self.look_over(p[0], p[1], p[2], 'above the TurtleBot marker', oblique=False,
                               view=self.view_height(rb) if rb else None)
            except Stop as e:
                if 'no look-down pose' not in str(e):
                    raise
                self.say('TurtleBot marker: no straight-down view there, measuring from here')
                res = self.bot_frames(a.frames)
                break
        if len(res) < need:
            raise Stop(f'TurtleBot marker ID {a.bot_id} lost')
        P = np.array([r['p'] for r in res])
        p = np.median(P, axis=0)
        yaws = np.array([r['yaw'] for r in res])
        yaw = math.atan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))
        tilt = float(np.median([r['tilt'] for r in res]))
        spread = float(np.max(np.linalg.norm(P[:, :2] - p[:2], axis=1)))
        if spread > 2 * a.max_spread:
            raise Stop(f'TurtleBot marker not stable ({spread * 1000:.1f} mm spread): moving?')
        if tilt > a.max_tilt:
            raise Stop(f'TurtleBot marker tilted {tilt:.0f} deg (> --max-tilt): not flat?')
        h = self.heading_marker() if a.go else None
        self.bot_how = f'BOX ID{a.bot_id}'
        if h is not None:
            d, dist = self.heading_side(p, yaw, h)
            self.bot_how += (f' + HEADING ID{a.bot_heading_id} OK'
                             if d >= 0.5 and 0.05 <= dist <= 0.40
                             else f' (HEADING ID{a.bot_heading_id} disagrees!)')
            if d < 0.5 or not 0.05 <= dist <= 0.40:
                self.say(f'WARNING: Burger heading marker ID {a.bot_heading_id} is not on the '
                         f'Burger side of the box marker (dot {d:+.2f}, {dist * 1000:.0f} mm): '
                         '--bot-side / marker places? (the box marker is used)')
            else:
                self.say(f'Burger heading marker ID {a.bot_heading_id}: agrees with the box '
                         f'marker (dot {d:+.2f}, {dist * 1000:.0f} mm)')
        self.bot = self.bot_from_marker(p, yaw, tilt)
        self.log_detection('bot', res)
        self.save_bot()
        self.bot_to_scene()
        return self.bot

    def bot_keepout(self, on):
        """'bot_keepout' in the planning scene: the TurtleBot area (regions.yaml, grown by
        --bot-keepout-margin) from the floor up to --bot-keepout-h over it (Burger with its
        LiDAR on top + clearance). on=False removes it."""
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from moveit_msgs.srv import ApplyPlanningScene
        from mycobot_calibration.moveit_client import _wait
        from mycobot_tasks import place_scene as ps
        a = self.args
        r = self.region('bot')
        if on and r is None:
            return False
        req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
        if on:
            m = a.bot_keepout_margin
            floor_z = getattr(self, 'floor_z', -0.306)
            (x0, x1), (y0, y1) = r['x'], r['y']
            objs = {'bot_keepout': [([(x0 + x1) / 2, (y0 + y1) / 2,
                                      floor_z + a.bot_keepout_h / 2],
                                     [x1 - x0 + 2 * m, y1 - y0 + 2 * m, a.bot_keepout_h], 0.0)]}
            req.scene.world.collision_objects = ps.collision_objects(objs, 'base')
        else:
            co = CollisionObject()
            co.header.frame_id, co.id, co.operation = 'base', 'bot_keepout', \
                CollisionObject.REMOVE
            req.scene.world.collision_objects = [co]
        res = _wait(self.mv.scene.call_async(req), 5.0)
        return res is not None and res.success

    @contextlib.contextmanager
    def bot_search(self, z=None):
        """While the TurtleBot pose is not known (finding its marker, or the tray on it with
        the marker hidden): the whole TurtleBot area is an obstacle up to --bot-keepout-h
        over the floor (bot_keepout: no motion goes through the Burger / its LiDAR), and
        every camera view over it is taken at least so high that the gripper tip (below
        the camera, looking down) stays over that: z = the surface looked at (default the
        bot region z)."""
        a = self.args
        if getattr(self, '_bot_search_on', False):   # nested (heading marker scan): the
            yield a.scan_height                      # outer keep-out stays as it is
            return
        r = self.region('bot')
        if z is None:
            z = r['z'] if r else -0.280
        floor_z = getattr(self, 'floor_z', -0.306)
        top = floor_z + a.bot_keepout_h
        # camera -> gripper tip along the view (looking straight down)
        X, T = self.node.T('link6', CAM), self.node.T('link6', TCP)
        cam_tip = float(T[2, 3] - X[2, 3]) if X is not None and T is not None else 0.16
        need = top + cam_tip + 0.01 - z
        view = max(need, float(r['view']) if r else 0.0)
        keep = (a.scan_height, a.search_height, a.search_min_height)
        added = False
        self._bot_search_on = True
        try:
            a.scan_height = a.search_height = a.search_min_height = view
            T_tip = self.node.T('base', TCP)
            inside = False
            if T_tip is not None and r is not None:
                m, tp = a.bot_keepout_margin, T_tip[:3, 3]
                inside = (r['x'][0] - m <= tp[0] <= r['x'][1] + m and
                          r['y'][0] - m <= tp[1] <= r['y'][1] + m and tp[2] < top)
            if inside:      # the arm would start in collision: no motion could be planned
                self.say('gripper tip inside the TurtleBot area now: kept-out block NOT added '
                         '(only the camera heights are raised)')
            elif a.bot_scene and self.use_scene:
                added = self.bot_keepout(True)
            self.say(f'TurtleBot area kept out up to {(top - floor_z) * 1000:.0f} mm over the '
                     f'floor (Burger + LiDAR){"" if added else " (NOT in the scene)"}: camera '
                     f'{view * 100:.0f} cm over z {z * 1000:.0f} mm, gripper tip >= '
                     f'{(top + 0.01 - floor_z) * 1000:.0f} mm over the floor')
            yield view
        finally:
            self._bot_search_on = False
            a.scan_height, a.search_height, a.search_min_height = keep
            if added:
                self.bot_keepout(False)

    def heading_marker(self, n=5, search=False):
        """The Burger heading marker (--bot-heading-id, --bot-heading-size, on the back of
        the LiDAR, user 11-26) -> its median xyz in base, or None. Seen from here; else the
        one seen within --heading-keep-s (same measurement, the Burger does not move); else
        (search, --go) the TurtleBot region scanned for it. Only its position is used, so it
        may stand upright."""
        a = self.args
        if getattr(a, 'bot_heading_id', -1) is None or a.bot_heading_id < 0:
            return None
        mid, size = a.bot_heading_id, a.bot_heading_size

        def look(k=n):
            res = self.bot_frames(k, mid=mid, size=size, timeout=2.0)
            if not res:
                return None
            h = np.median(np.array([r['p'] for r in res]), axis=0)
            self._heading_last = (time.monotonic(), h)
            return h

        h = look()
        if h is not None:
            return h
        last = getattr(self, '_heading_last', None)
        if last is not None and time.monotonic() - last[0] < a.heading_keep_s:
            self.say(f'Burger heading marker ID {mid} not in this view: the one seen '
                     f'{time.monotonic() - last[0]:.0f} s ago is used')
            return last[1]
        if not (search and a.go):
            return None
        self.say(f'Burger heading marker ID {mid} not seen: scanning the TurtleBot area for it')
        try:
            # Burger pose unknown (or from this marker): the TurtleBot area kept out
            with self.bot_search():
                self.scan_region('bot', lambda: len(self.bot_frames(2, mid=mid, size=size,
                                                                      timeout=1.5)) > 0)
        except Stop as e:
            if self._abort.is_set():
                raise
            self.say(f'Burger heading marker ID {mid}: {str(e).splitlines()[0]}')
            return None
        return look()

    def heading_side(self, box_xy, yaw, h):
        """Which way the Burger is, from its heading marker h (xyz) seen beside the box at
        box_xy with the marker +x at yaw: (dot, dist) of the box -> Burger direction of this
        yaw (--bot-side) with box -> h (dot > 0 = this yaw; dot < 0 = turned 180)."""
        sx, sy = self._bot_side_vec()
        c, s_ = math.cos(yaw), math.sin(yaw)
        b = np.array([c * sx - s_ * sy, s_ * sx + c * sy])      # box -> Burger, base frame
        v = np.asarray(h[:2], float) - np.asarray(box_xy[:2], float)
        dist = float(np.linalg.norm(v))
        return (float(b @ v) / dist if dist > 1e-6 else 0.0), dist

    def _bot_side_vec(self):
        side = '+y'
        sa = (self.args.scene_args or '').replace('=', ' ').split()
        if '--bot-side' in sa and sa.index('--bot-side') + 1 < len(sa):
            side = sa[sa.index('--bot-side') + 1]
        return {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}.get(side, (0, 1))

    def bot_from_tray(self, pin):
        """Tray in the TurtleBot box (box marker hidden under it) -> self.bot: the box at
        the tray (handle = marker + --bot-place-offset, box floor = handle top -
        --tray-height); the Burger direction from its heading marker (--bot-heading-id on
        the back of the LiDAR: the Burger lies from the box toward it, --bot-side of the box
        marker), looked for (the TurtleBot area scanned) when not in view. NOT from the tray
        direction (user 11-26: the tray may lie turned in the box) nor the last saved
        detection (the Burger is moved between runs). Heading marker not found -> Stop."""
        a = self.args
        h = self.heading_marker(search=True)
        if h is None:
            raise Stop(f'TurtleBot from the tray: Burger heading marker ID {a.bot_heading_id} '
                       'not found (TurtleBot area scanned): its direction is unknown, nothing '
                       'planned (is the marker on the LiDAR visible from above?)')
        v = np.asarray(h[:2], float) - np.asarray(pin['p'][:2], float)
        dist = float(np.linalg.norm(v))
        if not 0.05 <= dist <= 0.40:
            raise Stop(f'TurtleBot from the tray: heading marker ID {a.bot_heading_id} at '
                       f'({h[0] * 1000:.0f}, {h[1] * 1000:.0f}) mm is {dist * 1000:.0f} mm from the '
                       'tray (50..400 mm expected): another marker / Burger moved? nothing planned')
        sx, sy = self._bot_side_vec()
        yaw = math.remainder(math.atan2(v[1], v[0]) - math.atan2(sy, sx), 2 * math.pi)
        # info only: the tray in the box vs the Burger (not used)
        t = math.remainder(pin['yaw'] - math.radians(a.bot_yaw_offset) - yaw, math.pi)
        self.bot_how = f'HEADING ID{a.bot_heading_id} (tray not used)'
        self.say(f'Burger heading marker ID {a.bot_heading_id} at ({h[0] * 1000:.0f}, '
                 f'{h[1] * 1000:.0f}) mm, {dist * 1000:.0f} mm from the tray: Burger that way '
                 f'(box marker +x {math.degrees(yaw):.0f} deg; the tray lies {math.degrees(t):+.0f} '
                 'deg off the box axis, not used)')
        off = list(a.bot_place_offset) + [0.0] * (3 - len(a.bot_place_offset))
        c, s_ = math.cos(yaw), math.sin(yaw)
        hxy = np.asarray(pin['p'][:2], float)
        if a.bot_offset_frame == 'marker':
            mxy = hxy - np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
        else:
            mxy = hxy - np.array(off[:2])
        mz = float(pin['p'][2]) - a.tray_height + a.bot_marker_dz - off[2]
        self.say('TurtleBot pose from the tray position + heading marker (box marker under the tray):')
        return self.bot_from_marker([mxy[0], mxy[1], mz], yaw, float(pin.get('tilt', 0.0)))

    def bot_from_marker(self, p, yaw, tilt=0.0):
        """TurtleBot marker pose -> self.bot dict with the place spot / surface / tray
        direction from the --bot-* options."""
        a = self.args
        p = np.asarray(p, float)
        c, s_ = math.cos(yaw), math.sin(yaw)
        off = list(a.bot_place_offset) + [0.0] * (3 - len(a.bot_place_offset))
        dx, dy, dz = off[:3]
        if a.bot_offset_frame == 'marker':       # along the marker's own x / y axes
            place_xy = p[:2] + np.array([c * dx - s_ * dy, s_ * dx + c * dy])
        else:                                    # along base x / y
            place_xy = p[:2] + np.array([dx, dy])
        place_z = float(p[2]) - a.bot_marker_dz + dz
        place_yaw = math.remainder(yaw + math.radians(a.bot_yaw_offset), 2 * math.pi)
        self.say(f'TurtleBot marker ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, {p[2] * 1000:.0f}) '
                 f'mm, +x yaw {math.degrees(yaw):.0f} deg, tilt {tilt:.1f} deg -> place at '
                 f'({place_xy[0] * 1000:.0f}, {place_xy[1] * 1000:.0f}) mm, surface z '
                 f'{place_z * 1000:.0f} mm, tray L->R {math.degrees(place_yaw):.0f} deg '
                 f'(--bot-place-offset {dx * 1000:+.0f},{dy * 1000:+.0f},{dz * 1000:+.0f} mm in '
                 f'the {a.bot_offset_frame} frame)')
        return dict(p=p, yaw=yaw, tilt=tilt, place_xy=place_xy, place_z=place_z,
                    place_yaw=place_yaw)

    def bot_to_scene(self):
        """TurtleBot + box (walls --bot-rim) into the planning scene at self.bot."""
        a, b = self.args, self.bot
        if not (a.bot_scene and self.use_scene and b is not None):
            return
        c, s_ = math.cos(b['yaw']), math.sin(b['yaw'])
        bx, by = b['p'][:2] + np.array([c * a.bot_box_offset[0] - s_ * a.bot_box_offset[1],
                                        s_ * a.bot_box_offset[0] + c * a.bot_box_offset[1]])
        floor_z = getattr(self, 'floor_z', -0.306)
        self.bot_extra = ['--with-bot', f'--bot={bx:.4f},{by:.4f},{math.degrees(b["yaw"]):.1f}',
                          f'--box-floor={b["place_z"] - floor_z:.4f}',
                          f'--box-depth={max(0.0, a.bot_rim):.4f}']   # walls = --bot-rim
        self.apply_scene()

    def save_bot(self):
        import yaml
        b = self.bot
        d = dict(stamp=time.strftime('%Y-%m-%d %H:%M:%S'), marker_id=self.args.bot_id,
                 marker_xyz=[float(v) for v in b['p']], yaw_deg=math.degrees(b['yaw']),
                 tilt_deg=b['tilt'])
        try:
            with open(os.path.expanduser(self.args.bot_save), 'w') as f:
                yaml.safe_dump(d, f, sort_keys=False)
        except OSError as e:
            print(f'  TurtleBot detection not saved ({e})', flush=True)

    def measure_tray_on_bot(self):
        """The tray in the TurtleBot box: looked for at the bot region height (regions.yaml
        z) with the TurtleBot area kept out (its marker lies under the tray), measured ->
        pin; --table-z = the box floor from the measured handle top, heights set, and
        self.bot (+ scene) from the tray (bot_from_tray). The caller restores --table-z."""
        a = self.args
        rb = self.region('bot')
        z0 = rb['z'] if rb else (self.bot['place_z'] if self.bot else None)
        if z0 is None:
            raise Stop(f'no TurtleBot area in {a.regions} (its z = the box floor): run '
                       '`ros2 run mycobot_tasks regions`')
        a.table_z = z0                        # the tray stands in the TurtleBot box
        with self.bot_search(z0):        # Burger / LiDAR pose unknown: kept out
            if a.go and self.bot is not None:     # found this session: look there first
                try:
                    self.look_over(self.bot['place_xy'][0], self.bot['place_xy'][1], z0,
                                   'over the TurtleBot',
                                   view=self.view_height(rb) if rb else None)
                except Stop as e:
                    if 'no look-down pose' not in str(e) and 'motion failed' not in str(e):
                        raise
                    print(f'  {e}: detecting from here', flush=True)
            self.expect_on_bot = True
            # measured again from closer (--bot-close-view): this first one only for the
            # position (tilt / spacing checked on the close one)
            self.detect_provisional = bool(a.go and a.bot_close_view > 0)
            try:
                pin = self.detect_pin(region='bot')
            finally:
                self.expect_on_bot = False
                self.detect_provisional = False
            # box floor from the measured handle top, TurtleBot (scene) from the tray
            a.table_z = float(pin['p'][2]) - a.tray_height
            self.set_heights(a.tray_height)
            self.bot = self.bot_from_tray(pin)
            self.save_bot()
            self.bot_to_scene()
        return self._tray_on_bot_closer(pin)

    def _tray_on_bot_closer(self, pin):
        """The search view over the TurtleBot is high (the gripper kept over the Burger /
        LiDAR while their pose is unknown): small markers, a poor tilt. Now the Burger is in
        the scene (from the tray): the camera --bot-close-view over the tray and measured
        again; no pose / not seen there -> the first measurement."""
        a = self.args
        if not a.go or a.bot_close_view <= 0:
            return pin
        prov = pin.get('provisional', False)   # first view unchecked: no fallback to it
        z = float(pin['p'][2]) - a.tray_height
        try:
            self.look_over(float(pin['p'][0]), float(pin['p'][1]), z,
                           'closer over the tray in the TurtleBot box', view=a.bot_close_view)
        except Stop as e:
            if self._abort.is_set() or prov:
                raise Stop(f'closer view over the TurtleBot tray not possible: {e}')
            self.say(f'closer view over the TurtleBot tray not possible ({str(e).splitlines()[0]}): '
                     'the first measurement is used')
            return pin
        self.expect_on_bot = True
        try:
            pin2 = self.detect_pin(region='bot')
        except Stop as e:
            if self._abort.is_set() or prov:
                raise Stop(f'closer view: {e}')
            self.say(f'closer view: tray not measured ({str(e).splitlines()[0]}): the first '
                     'measurement is used')
            return pin
        finally:
            self.expect_on_bot = False
        d = float(np.linalg.norm(np.asarray(pin2['p'][:2]) - np.asarray(pin['p'][:2]))) * 1000
        self.say(f'tray in the TurtleBot box measured again from {a.bot_close_view * 100:.0f} cm: '
                 f'{d:.1f} mm from the first measurement')
        a.table_z = float(pin2['p'][2]) - a.tray_height
        self.set_heights(a.tray_height)
        self.bot = self.bot_from_tray(pin2)
        self.save_bot()
        self.bot_to_scene()
        return pin2
