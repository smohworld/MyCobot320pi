"""TurtleBot: marker detection -> place spot / height / tray direction, its planning
scene objects, and r (the tray from the TurtleBot back to where m picked it)."""
import math
import os
import time

import cv2
import numpy as np

from mycobot_tasks.red_cube_hover import CAM, Stop, project


class BotMixin:
    def bot_frames(self, n):
        """n fresh frames -> TurtleBot marker (--bot-id, --bot-marker-size) results:
        dict(p centre xyz base, yaw of the marker +x, normal, tilt, px, off, view)."""
        a = self.args
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
            self.move('home', home)

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
        with the original L -> R direction -> home."""
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
        if self.bot is None:
            self.bot = self.load_bot()
        b = self.bot
        if b is None:
            raise Stop('TurtleBot position unknown: t first')
        if not self.bot_extra:
            self.bot_to_scene()
        home = self.home
        oxy = np.array(o['handle_xyz'][:2], float)
        keep = (a.table_z, a.place_z, a.place_at)
        try:
            a.table_z = b['place_z']              # the tray stands on the TurtleBot now
            rb = self.region('bot')
            self.say(f"put back: tray on the TurtleBot ({b['place_xy'][0] * 1000:.0f}, "
                     f"{b['place_xy'][1] * 1000:.0f}) mm -> origin ({oxy[0] * 1000:.0f}, "
                     f"{oxy[1] * 1000:.0f}) mm, surface z {o['surface_z'] * 1000:.0f} mm")
            if a.go:
                try:
                    self.look_over(b['place_xy'][0], b['place_xy'][1], b['place_z'],
                                   'over the TurtleBot',
                                   view=self.view_height(rb) if rb else None)
                except Stop as e:
                    if 'no look-down pose' not in str(e) and 'motion failed' not in str(e):
                        raise
                    print(f'  {e}: detecting from here', flush=True)
            pin = self.detect_pin(region='bot')
            xy = pin['p'][:2].copy()
            a.place_z, a.place_at = float(o['surface_z']), [float(oxy[0]), float(oxy[1])]
            turn = None
            if not a.return_keep_yaw:              # back to the original L -> R direction
                turn = math.remainder(math.radians(o['yaw_lr_deg']) - pin['yaw'], 2 * math.pi)
            plan = self.plan_pick(xy, pin['yaw'] + math.radians(a.grip_yaw_offset), oxy, turn,
                                  rim=0.0)
            self.say(f'put back plan OK: ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) -> '
                     f'({oxy[0] * 1000:.0f}, {oxy[1] * 1000:.0f}) mm, tray L->R '
                     f"{math.degrees(pin['yaw']):.0f} -> "
                     f"{o['yaw_lr_deg'] if turn is not None else math.degrees(pin['yaw']):.0f}"
                     ' deg, then home')
            self.transfer(pin, plan, oxy, home)
        finally:
            a.table_z, a.place_z, a.place_at = keep
        o['returned'] = True
        try:
            import yaml
            with open(os.path.expanduser(a.origin_save), 'w') as f:
                yaml.safe_dump(o, f, sort_keys=False)
        except OSError:
            pass
