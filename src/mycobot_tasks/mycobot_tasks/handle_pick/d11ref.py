"""D11 target without a shelf marker: the tray's own markers measured while it stands on D11
(registration --do d11ref, every n before picking, every j after placing) are kept in
--d11-ref (median of the last --d11-ref-n); j puts the tray there. The layout export
(--relay-d11, model) is only the fallback / first camera look."""
import math
import os
import time

import numpy as np

from mycobot_tasks.red_cube_hover import Stop

KEEP = 20          # samples kept in the file


class D11RefMixin:
    # ------------------------------------------------------------------ file
    def d11_ref_load(self):
        """-> dict(h, lr, n, last, registered, samples) from --d11-ref, or None."""
        import yaml
        path = os.path.expanduser(self.args.d11_ref or '')
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                d = yaml.safe_load(f) or {}
            s = [list(v) for v in d.get('samples') or []]
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            return None
        if not s:
            return None
        use = s[-max(1, int(self.args.d11_ref_n)):]
        xyz = np.median(np.array([v[:3] for v in use], float), axis=0)
        yaws = np.radians([float(v[3]) for v in use])
        lr = math.atan2(float(np.median(np.sin(yaws))), float(np.median(np.cos(yaws))))
        return dict(h=xyz, lr=lr, n=len(use), last=str(use[-1][4]),
                    registered=str(d.get('registered', '')), samples=s)

    def _d11_ref_save(self, samples, registered):
        import yaml
        path = os.path.expanduser(self.args.d11_ref)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            f.write('# tray on D11 as measured (tray markers; no shelf marker). Written by\n'
                    '# tray_handle_pick: --do d11ref (registration), n (before picking),\n'
                    '# j (after placing). samples: x, y, z [m] handle top, L->R [deg], time, source\n')
            yaml.safe_dump(dict(registered=registered, samples=samples[-KEEP:]), f,
                           default_flow_style=None, sort_keys=False)
        os.replace(tmp, path)

    @staticmethod
    def _sample(pin, source):
        p = [round(float(v), 4) for v in pin['p'][:3]]
        return p + [round(math.degrees(float(pin['yaw'])), 2),
                    time.strftime('%Y-%m-%d %H:%M:%S'), source]

    @staticmethod
    def _d11_diff(h, lr, ref):
        d = np.asarray(h[:2], float) - np.asarray(ref['h'][:2], float)
        dz = float(h[2]) - float(ref['h'][2])
        dyaw = math.degrees(math.remainder(lr - ref['lr'], math.pi))
        return d, dz, dyaw

    def d11_ref_add(self, pin, source):
        """A measurement of the tray standing on D11 (real robot) -> into --d11-ref, unless it
        is more than --d11-ref-warn off the reference (then WARNING, not added: a tray not
        pushed in / a misread must not move the target; the shelf moved -> --do d11ref)."""
        a = self.args
        if not a.d11_ref or a.layout_poses or self.conn in ('fake', None):
            return
        ref = self.d11_ref_load()
        h, lr = np.asarray(pin['p'], float), float(pin['yaw'])
        if ref is not None:
            d, dz, dyaw = self._d11_diff(h, lr, ref)
            dist = float(np.linalg.norm(d))
            wz = a.d11_ref_warn[2] if len(a.d11_ref_warn) > 2 else None
            if dist > a.d11_ref_warn[0] or abs(dyaw) > a.d11_ref_warn[1] or \
                    (wz is not None and abs(dz) > wz):
                self.say(f'WARNING: tray on D11 ({source}) measured {dist * 1000:.1f} mm '
                         f'(dx {d[0] * 1000:+.1f}, dy {d[1] * 1000:+.1f}, dz {dz * 1000:+.1f}), '
                         f'yaw {dyaw:+.1f} deg off the D11 reference: NOT added (tray not '
                         'pushed in, misread, or the shelf moved -> --do d11ref)')
                return
            samples = ref['samples']
        else:
            samples = []
        samples.append(self._sample(pin, source))
        try:
            self._d11_ref_save(samples, ref['registered'] if ref else '')
        except OSError as e:
            self.say(f'D11 reference not written ({e})')
            return
        new = self.d11_ref_load()
        self.say(f'D11 reference updated ({source}): handle ({new["h"][0] * 1000:.1f}, '
                 f'{new["h"][1] * 1000:.1f}, {new["h"][2] * 1000:.1f}) mm, L->R '
                 f'{math.degrees(new["lr"]):.1f} deg (median of {new["n"]})')

    # ------------------------------------------------------------------ target
    def d11_target(self):
        """Where the tray goes on D11 / where to look first: the measured reference
        (--d11-ref), else the layout export (--relay-d11, model) with a WARNING.
        -> (dict(h, lr), source text)."""
        a = self.args
        ref = self.d11_ref_load()
        if ref is not None:
            src = f'D11 reference ({a.d11_ref}, median of {ref["n"]}, last {ref["last"]})'
            try:
                age = (time.time() - time.mktime(time.strptime(ref['last'],
                                                               '%Y-%m-%d %H:%M:%S'))) / 86400
            except ValueError:
                age = 0.0
            if a.d11_ref_age_days and age > a.d11_ref_age_days:
                self.say(f'WARNING: the D11 reference is {age:.0f} days old (no n / j / d11ref '
                         'since): if the shelf moved, run --do d11ref')
            try:
                model = self.relay_d11_file()
                d, dz, dyaw = self._d11_diff(ref['h'], ref['lr'], model)
                src += (f'; vs the model {np.linalg.norm(d) * 1000:.0f} mm, dz '
                        f'{dz * 1000:+.0f} mm, yaw {dyaw:+.1f} deg')
            except (OSError, KeyError, TypeError, ValueError):
                pass
            return dict(h=ref['h'], lr=ref['lr']), src
        d = self.relay_d11_file()
        if not (a.layout_poses or self.conn in ('fake', None)):
            self.say('WARNING: no D11 reference (--d11-ref): the layout MODEL pose is used for '
                     'D11 (not measured; register once: --do d11ref with a tray pushed into D11)')
        return d, f'layout model ({a.relay_d11})'

    def sim_d11(self, d11):
        """--layout-poses: the D11 pose the planner gets = layout + --sim-d11-error (dx, dy
        [m], dyaw [deg], dz [m]): a measurement / reference error vs the shelf in the scene."""
        e = list(self.args.sim_d11_error or [])
        if not e or not any(e):
            return d11
        e = (e + [0.0] * 4)[:4]
        h = np.array(d11['h'], float)
        h[:2] += e[:2]
        h[2] += e[3]
        lr = float(d11['lr']) + math.radians(e[2])
        self.say(f'(simulation) D11 pose given to the planner {e[0] * 1000:+.0f}, '
                 f'{e[1] * 1000:+.0f} mm, {e[2]:+.1f} deg, dz {e[3] * 1000:+.0f} mm off the '
                 'shelf (--sim-d11-error)')
        return dict(h=h, lr=lr)

    def _look_d11(self, d11):
        a = self.args
        self.look_last((d11['h'][0], d11['h'][1], d11['h'][2] - a.tray_height), 'shelf')

    def d11_empty_check(self, d11):
        """j, real robot: camera over D11 first; a tray seen within 5 cm of the target -> Stop
        (D11 is not empty). Nothing seen = empty."""
        a = self.args
        if not a.d11_empty_check or not a.go or a.layout_poses or self.conn in ('fake', None):
            return
        self.say('D11 must be empty: camera over D11')
        keep = (a.search_x, a.search_y, a.use_last, a.table_z)
        try:
            a.search_x, a.search_y, a.use_last = float(d11['h'][0]), float(d11['h'][1]), False
            a.table_z = float(d11['h'][2]) - a.tray_height
            self._look_d11(d11)
            try:
                pin = self.detect_pin(region='desk')     # no such region: no wide scan
            except Stop as e:
                if self._abort.is_set():
                    raise
                self.say(f'D11 empty (no tray seen: {str(e).splitlines()[0]})')
                return
        finally:
            a.search_x, a.search_y, a.use_last, a.table_z = keep
        dist = float(np.linalg.norm(np.asarray(pin['p'][:2]) - np.asarray(d11['h'][:2])))
        if dist < 0.05:
            raise Stop(f'D11 is not empty: tray {a.tray} seen {dist * 1000:.0f} mm from the '
                       'D11 target')
        self.say(f'D11 empty (a tray seen {dist * 1000:.0f} mm away: not on D11)')

    # ------------------------------------------------------------------ registration
    def act_d11_ref(self):
        """--do d11ref (real robot): a tray pushed fully into D11 by hand is measured
        --d11-ref-samples times; the median becomes the D11 reference (old samples
        replaced). Compared with the layout model and the previous reference."""
        a = self.args
        if a.layout_poses or self.conn in ('fake', None):
            raise Stop('d11ref: real robot only (the simulation has the layout D11)')
        if self.holding:
            raise Stop('holding the tray: put it down first')
        if not a.d11_ref:
            raise Stop('d11ref: --d11-ref is empty')
        old = self.d11_ref_load()
        guess, src = self.d11_target()
        self.say(f'D11 registration: camera over D11 ({src})')
        keep = (a.search_x, a.search_y, a.use_last, a.table_z)
        pins = []
        try:
            a.search_x, a.search_y, a.use_last = float(guess['h'][0]), float(guess['h'][1]), False
            a.table_z = float(guess['h'][2]) - a.tray_height
            if a.go:
                self._look_d11(guess)
            for i in range(max(1, int(a.d11_ref_samples))):
                pin = self.detect_pin(region='shelf')
                self.check_tray()
                pins.append(pin)
                self.say(f'  {i + 1}/{a.d11_ref_samples}: ({pin["p"][0] * 1000:.1f}, '
                         f'{pin["p"][1] * 1000:.1f}, {pin["p"][2] * 1000:.1f}) mm, L->R '
                         f'{math.degrees(float(pin["yaw"])):.1f} deg')
                time.sleep(0.3)
        finally:
            a.search_x, a.search_y, a.use_last, a.table_z = keep
        P = np.array([p['p'][:3] for p in pins], float)
        med = np.median(P, axis=0)
        spread = float(np.max(np.linalg.norm(P[:, :2] - med[:2], axis=1)))
        samples = [self._sample(p, 'd11ref') for p in pins]
        self._d11_ref_save(samples, time.strftime('%Y-%m-%d %H:%M:%S'))
        new = self.d11_ref_load()
        self.say(f'D11 reference registered: handle ({new["h"][0] * 1000:.1f}, '
                 f'{new["h"][1] * 1000:.1f}, {new["h"][2] * 1000:.1f}) mm, L->R '
                 f'{math.degrees(new["lr"]):.1f} deg, spread {spread * 1000:.1f} mm '
                 f'({len(pins)} measurements) -> {a.d11_ref}')
        if spread > a.d11_ref_warn[0] / 2:
            self.say(f'WARNING: the measurements spread {spread * 1000:.1f} mm: check the view '
                     '(markers small / oblique?) and register again')
        for name, ref in (('layout model', None), ('previous reference', old)):
            try:
                ref = ref if ref is not None else (self.relay_d11_file() if name.startswith('layout') else None)
            except (OSError, KeyError, TypeError, ValueError):
                ref = None
            if ref is None:
                continue
            d, dz, dyaw = self._d11_diff(new['h'], new['lr'], ref)
            self.say(f'  vs the {name}: dx {d[0] * 1000:+.1f}, dy {d[1] * 1000:+.1f}, dz '
                     f'{dz * 1000:+.1f} mm, yaw {dyaw:+.1f} deg')
            if name.startswith('layout'):
                self._fork_top_hint(dz)

    def _fork_top_hint(self, dz):
        """After a registration: the layout fork height (layout.yaml fork_top, used by the
        planning scene on the real robot too) that matches the measured D11 height."""
        import yaml
        path = os.path.expanduser('~/.ros/layout.yaml')
        try:
            with open(path) as f:
                ft = float(yaml.safe_load(f)['fork_top'])
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
            return
        new = ft + dz
        self.say(f'  fork top: layout {ft * 1000:.0f} mm over the floor, measured '
                 f'{new * 1000:.0f} mm' + (
                     f' -> set fork_top {new:.3f} in {path} (ros2 run mycobot_tasks layout, '
                     'D11 height) so the planning scene matches' if abs(dz) > 0.003 else
                     ' (matches within 3 mm)'))

