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
    def bot_frames(self, n):
        """n fresh frames -> TurtleBot marker (--bot-id, --bot-marker-size) results:
        dict(p centre xyz base, yaw of the marker +x, normal, tilt, px, off, view)."""
        a = self.args
        if self._settle_now in ('once', 'fast'):
            self.wait_still()      # the last (--mid-settle) motion may end while moving
        out, after, t0, got = [], time.monotonic(), time.monotonic(), 0
        while got < n and time.monotonic() - t0 < a.pin_timeout:
            img = self.node.fresh(after, timeout=1.0)
            if img is None:
                continue
            after = time.monotonic()
            got += 1
            T = self.node.T('base', CAM)
            if T is None:
                raise Stop(f'no TF base -> {CAM}')
            found = self.bot_mp.detect(img)
            if a.bot_id not in found:
                continue
            c, rv, tv, _ = found[a.bot_id]
            R = T[:3, :3] @ cv2.Rodrigues(rv)[0]
            p = T[:3, :3] @ tv.reshape(3) + T[:3, 3]
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
            a.scan_height, a.search_height, a.search_min_height = keep
            if added:
                self.bot_keepout(False)

    def bot_from_tray(self, pin):
        """Tray in the TurtleBot box (marker hidden under it) -> self.bot as if the marker
        had been seen: the inverse of the place (handle = marker + --bot-place-offset, tray
        L -> R = marker +x + --bot-yaw-offset, box floor = handle top - --tray-height).
        L -> R and R -> L give the same tray: of the two the marker yaw closer to the last
        saved TurtleBot detection is used (none saved: the tray's L -> R)."""
        a = self.args
        yaw = math.remainder(pin['yaw'] - math.radians(a.bot_yaw_offset), 2 * math.pi)
        last = None
        try:
            import yaml
            with open(os.path.expanduser(a.bot_save)) as f:
                last = math.radians(float(yaml.safe_load(f)['yaw_deg']))
        except (OSError, KeyError, TypeError, ValueError, ImportError):
            pass
        if last is not None and abs(math.remainder(yaw - last, 2 * math.pi)) > math.pi / 2:
            yaw = math.remainder(yaw + math.pi, 2 * math.pi)
        off = list(a.bot_place_offset) + [0.0] * (3 - len(a.bot_place_offset))
        c, s_ = math.cos(yaw), math.sin(yaw)
        hxy = np.asarray(pin['p'][:2], float)
        if a.bot_offset_frame == 'marker':
            mxy = hxy - np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
        else:
            mxy = hxy - np.array(off[:2])
        mz = float(pin['p'][2]) - a.tray_height + a.bot_marker_dz - off[2]
        self.say('TurtleBot pose from the tray (its marker is under it):')
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

    def load_bot(self):
        """The last TurtleBot detection (--bot-save) -> self.bot, or None."""
        import yaml
        try:
            with open(os.path.expanduser(self.args.bot_save)) as f:
                d = yaml.safe_load(f)
            print(f"TurtleBot from {self.args.bot_save} ({d['stamp']})", flush=True)
            return self.bot_from_marker(d['marker_xyz'], math.radians(d['yaw_deg']),
                                        float(d.get('tilt_deg', 0.0)))
        except (OSError, KeyError, TypeError, ValueError):
            return None

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

    def act_find_bot(self):
        """t: find the TurtleBot (and back to home with --bot-home)."""
        home = self.home
        self.find_bot()
        if self.args.go and self.args.bot_home and home is not None:
            self.move('home', home, mid=True)

    def wants_bot(self):
        a = self.args
        return a.place_at == 'bot' or a.place_z_spec == 'auto' or a.place_yaw_spec == 'bot'

    def save_origin(self, pin, surface_z):
        import yaml
        d = dict(stamp=time.strftime('%Y-%m-%d %H:%M:%S'), tray=self.args.tray,
                 handle_xyz=[float(v) for v in pin['p']], yaw_lr_deg=math.degrees(pin['yaw']),
                 surface_z=float(surface_z), returned=False)
        try:
            with open(os.path.expanduser(self.args.origin_save), 'w') as f:
                yaml.safe_dump(d, f, sort_keys=False)
            print(f'  origin saved: {self.args.origin_save} (r = put it back there)', flush=True)
        except OSError as e:
            print(f'  origin not saved ({e})', flush=True)

    def load_origin(self):
        import yaml
        try:
            with open(os.path.expanduser(self.args.origin_save)) as f:
                return yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            return None

    def act_return(self):
        """r: the tray from the TurtleBot back to where m picked it (--origin-save):
        look at the TurtleBot spot, detect the handle there (nothing seen -> TurtleBot region
        scan), plan everything, pick -> lift -> carry -> put down at the origin xy / surface
        with the original L -> R direction (no home after)."""
        a = self.args
        if self.holding:
            raise Stop('holding the tray: put it down first')
        o = self.load_origin()
        if o is None:
            raise Stop(f'no origin in {a.origin_save}: m (pick + place on the TurtleBot) '
                       'saves it')
        if o.get('returned') and not a.return_again:
            raise Stop(f"the tray from {o['stamp']} was already put back (--return-again "
                       'to do it anyway)')
        oxy = np.array(o['handle_xyz'][:2], float)
        self.from_bot(oxy, float(o['surface_z']),
                      None if a.return_keep_yaw else o['yaw_lr_deg'], 'origin')
        o['returned'] = True
        try:
            import yaml
            with open(os.path.expanduser(a.origin_save), 'w') as f:
                yaml.safe_dump(o, f, sort_keys=False)
        except OSError:
            pass

    def shelf_spot(self):
        """Place spot on the shelf (no marker there): the centre of the shelf region
        (regions.yaml, set in RViz) -> (xy, surface z)."""
        r = self.region('shelf')
        if r is None:
            raise Stop(f'no shelf area in {self.args.regions}: run '
                       '`ros2 run mycobot_tasks regions` and set it in RViz')
        return np.array([sum(r['x']) / 2, sum(r['y']) / 2]), float(r['z'])

    def act_bot_to_shelf(self):
        """k: the tray from the TurtleBot onto the shelf: detect it on the TurtleBot, plan
        everything, pick -> lift -> carry -> put down with the handle midpoint at the shelf
        region centre (--shelf-yaw, else a numeric --place-yaw: tray L -> R direction
        there; default: kept)."""
        a = self.args
        if self.holding:
            raise Stop('holding the tray: put it down first')
        xy, z = self.shelf_spot()
        # only a number given for --place-yaw ('bot' = the TurtleBot direction, set by m)
        yaw = None if a.place_yaw_spec in (None, 'bot') else float(a.place_yaw_spec)
        if a.shelf_yaw is not None:           # --shelf-yaw: the shelf direction on its own
            yaw = a.shelf_yaw
        self.from_bot(xy, z, yaw, 'shelf centre')

    def from_bot(self, oxy, oz, yaw_deg, label):
        """Tray on the TurtleBot -> put down with the handle midpoint at oxy on the surface
        oz [m], tray L -> R yaw_deg (None = as it is on the TurtleBot).
        The TurtleBot marker lies under the tray (box floor centre): it cannot be seen now.
        The tray is looked for at the bot region height (regions.yaml z), and the TurtleBot
        pose for the planning scene comes from the tray (bot_from_tray)."""
        a = self.args
        rb = self.region('bot')
        z0 = rb['z'] if rb else (self.bot['place_z'] if self.bot else None)
        if z0 is None:
            raise Stop(f'no TurtleBot area in {a.regions} (its z = the box floor): run '
                       '`ros2 run mycobot_tasks regions`')
        keep = (a.table_z, a.place_z, a.place_at)
        try:
            a.table_z = z0                        # the tray stands in the TurtleBot box
            self.say(f'tray on the TurtleBot (box floor z {z0 * 1000:.0f} mm) -> {label} '
                     f'({oxy[0] * 1000:.0f}, {oxy[1] * 1000:.0f}) mm, surface z '
                     f'{oz * 1000:.0f} mm')
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
                try:
                    pin = self.detect_pin(region='bot')
                finally:
                    self.expect_on_bot = False
                xy = pin['p'][:2].copy()
                # box floor from the measured handle top, TurtleBot (scene) from the tray
                a.table_z = float(pin['p'][2]) - a.tray_height
                self.set_heights(a.tray_height)
                self.bot = self.bot_from_tray(pin)
                self.save_bot()
                self.bot_to_scene()
            a.place_z, a.place_at = oz, [float(oxy[0]), float(oxy[1])]
            turn = None
            if yaw_deg is not None:
                turn = math.remainder(math.radians(yaw_deg) - pin['yaw'], 2 * math.pi)
            plan = self.plan_pick(xy, pin['yaw'] + math.radians(a.grip_yaw_offset), oxy, turn,
                                  rim=0.0)
            self.say(f'TurtleBot -> {label} plan OK: ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) '
                     f'-> ({oxy[0] * 1000:.0f}, {oxy[1] * 1000:.0f}) mm, tray L->R '
                     f"{math.degrees(pin['yaw']):.0f} -> "
                     f"{yaw_deg if turn is not None else math.degrees(pin['yaw']):.0f} deg")
            self.transfer(pin, plan, oxy)
        finally:
            a.table_z, a.place_z, a.place_at = keep
