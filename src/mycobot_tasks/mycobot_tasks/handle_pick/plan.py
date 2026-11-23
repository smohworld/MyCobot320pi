"""Whole-motion planning before anything moves: straight legs checked with chained IK
(no IK branch flips), pick / carry / place legs, lowering the first pose when far out."""
import math

import numpy as np

from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, Stop, TCP


class PlanMixin:
    def line(self, p0, p1, ty, q0, ty1=None):
        """Straight TCP line p0 -> p1, tool pointing down, tool yaw ty -> ty1 (None = ty
        kept; turned the short way, as Pilz LIN does): chained IK every --step-check (and
        every 5 deg of turn) from q0; no step may move a joint more than --max-step-jump
        (else the arm flips between IK branches and a joint move would not follow the
        line). -> (q at p1, None) or (None, why)."""
        a = self.args
        memo = getattr(self, '_memo', None)
        if memo is not None:
            # one plan_pick tries many variants (tool yaw, IK branch, carry height, lowered
            # heights) that share legs: each distinct leg is solved once
            key = ('line', tuple(np.round(p0, 4)), tuple(np.round(p1, 4)), round(ty, 4),
                   None if ty1 is None else round(ty1, 4), tuple(np.round(q0, 3)),
                   self.plan_grip, a.step_check)
            if key not in memo:
                memo[key] = self._line(p0, p1, ty, q0, ty1)
            q, why = memo[key]
            return (None if q is None else q.copy()), why
        return self._line(p0, p1, ty, q0, ty1)

    def _line(self, p0, p1, ty, q0, ty1=None):
        a = self.args
        turn = 0.0 if ty1 is None else math.remainder(ty1 - ty, 2 * math.pi)
        n = max(1, int(math.ceil(float(np.linalg.norm(p1 - p0)) / a.step_check)),
                int(math.ceil(abs(math.degrees(turn)) / 5.0)))
        qs = q0
        for i in range(1, n + 1):
            self._check()
            pi = p0 + (p1 - p0) * i / n
            qi = self.ik_fixed(pi, ty + turn * i / n, qs)
            if qi is None:
                return None, (f'no IK at ({pi[0] * 1000:.0f}, {pi[1] * 1000:.0f}, '
                              f'{pi[2] * 1000:.0f}) mm: {self.ik_why(pi, ty, qs)}')
            dq = np.abs(qi - qs)
            st = math.degrees(float(np.max(dq)))
            if st > a.max_step_jump:
                j = int(np.argmax(dq))
                at = f'({pi[0] * 1000:.0f}, {pi[1] * 1000:.0f}, {pi[2] * 1000:.0f}) mm'
                # a big step right before the end of the line that cannot be reached at all:
                # the arm straightens (J3 -> 0, singular), not a branch flip
                if self.ik_fixed(p1, ty if ty1 is None else ty + turn, qs) is None:
                    return None, (f'reach limit: arm nearly straight (J3 '
                                  f'{math.degrees(qi[2]):.0f} deg) at {at}, the end '
                                  f'({p1[0] * 1000:.0f}, {p1[1] * 1000:.0f}, '
                                  f'{p1[2] * 1000:.0f}) mm r {math.hypot(p1[0], p1[1]) * 1000:.0f}'
                                  f' mm is out of reach: '
                                  f'{self.ik_why(p1, ty if ty1 is None else ty + turn, qs)}')
                return None, (f'J{j + 1} step {st:.0f} deg at {at} (IK branch flip'
                              + (', or J6 at its limit while turning' if turn else '') + ')')
            qs = qi
        return qs, None

    def plan_pick(self, xy, yaw, place_xy=None, turn=None, rim=None):
        """_plan_pick; when it fails, the first pose over the handle (servo height, also
        the lowest carry height) is lowered in 1 cm steps down to the approach height, then
        both together down to --min-approach-above over the handle top: a tray on a high
        surface far out (the TurtleBot) is reachable only from
        closer above."""
        a = self.args
        keep = a.servo_height, a.approach
        h = a.approach - a.approach_above                  # handle top over the surface
        # (above, approach) heights tried: above lowered to the approach height, then both
        # together down to --min-approach-above over the handle top
        tries = []
        s = keep[0]
        while s > keep[1] + 1e-9:
            tries.append((s, keep[1]))
            s -= 0.01
        s = keep[1]
        while s >= h + a.min_approach_above - 1e-9:
            tries.append((s, s))
            s -= 0.01
        first = last = None
        outer = getattr(self, '_memo', None) is None
        if outer:
            self._memo = {}
        try:
            for s, ap in tries:
                a.servo_height, a.approach = s, ap
                try:
                    plan = self._plan_pick(xy, yaw, place_xy, turn, rim)
                    if (s, ap) != keep:
                        self.say(f'  (out of reach higher: first pose {(s - h) * 1000:.0f} mm, '
                                 f'approach / lift {(ap - h) * 1000:.0f} mm over the handle top)')
                    return plan
                except Stop as e:
                    first = first or e
                    last = (s, ap, e)
            if last is None or len(tries) == 1:
                raise first
            s, ap, e = last
            raise Stop(f'{first}\n  ... also lowered down to first pose {(s - h) * 1000:.0f} mm / '
                       f'approach {(ap - h) * 1000:.0f} mm over the handle top, still failed:\n'
                       + str(e).split('\n', 1)[-1])
        finally:
            a.servo_height, a.approach = keep
            if outer:
                self._memo = None

    def leg_grip(self, name):
        """Gripper opening a leg is planned with: open only --grasp-open far around the
        handle (to it, after releasing), --hold-grip while the tray is held."""
        a = self.args
        if name in ('above', 'approach', 'grasp', 'p_lift', 'p_back'):
            return a.grasp_open
        return a.hold_grip                 # put, high, via*, p_high, p_put

    def _plan_pick(self, xy, yaw, place_xy=None, turn=None, rim=None):
        try:
            return self._plan_pick_(xy, yaw, place_xy, turn, rim)
        finally:
            self.plan_grip = None

    def _plan_pick_(self, xy, yaw, place_xy=None, turn=None, rim=None):
        """The whole motion at once, before anything moves (pick and put back; with
        place_xy also the transfer and the place there):
          above (h + --servo-above) -> approach (h + --approach-above) -> grasp
          (h - --grasp-depth) -> lift (= approach); put = grasp + --place-drop (put back)
          place: above -> high (straight up: tray bottom --carry-clear over the place
                 surface --place-z) -> p_high (straight across) -> p_put (down) -> open ->
                 p_lift (finger tips --approach-above over the handle)
        The tool yaw stays the same, except on the carry (p_high) when turn is given: the
        tray is turned by that much [rad] (from --place-yaw). Both pick yaws that give the
        same pinch (yaw, yaw + 180) are tried, the one with the least joint motion wins.
        Every leg after 'above' is checked as a straight line (line()).
        -> dict(tcp_yaw, legs={name: (p, q)}) or Stop with the failing leg and why."""
        a = self.args
        q_now = self.node.q()
        if q_now is None:
            raise Stop('no joint_states')
        x, y = float(xy[0]), float(xy[1])

        def P(px, py, h):
            return np.array([px, py, a.table_z + h])
        steps = [('above', P(x, y, a.servo_height), None),
                 ('approach', P(x, y, a.approach), 'above'),
                 ('grasp', P(x, y, a.grasp_z), 'approach'),
                 ('put', P(x, y, a.grasp_z + a.place_drop), 'grasp')]
        if place_xy is not None:
            # place surface (--place-z, e.g. the desk top) vs the pick surface (--table-z)
            # carry: the tray bottom (grasp height - grasp_z below the tip) --carry-clear
            # over the place surface, never lower than 'above'
            # heights at the place from --tray-height (tray bottom -> handle top), not from
            # the pick surface: a wrong --table-z must not drop the tray from the air
            u, v = float(place_xy[0]), float(place_xy[1])
            g_tip = a.tray_height - a.grasp_depth          # finger tip over the tray bottom
            # into the TurtleBot box: the tray bottom passes --carry-clear over its rim
            if rim is None:
                rim = a.bot_rim if a.place_at == 'bot' else 0.0
            carry0 = max(a.table_z + a.servo_height,
                         a.place_z + rim + g_tip + a.carry_clear)
            p_steps = [('p_put', np.array([u, v, a.place_z + g_tip + a.place_drop]), 'p_high'),
                       ('p_lift', np.array([u, v, a.place_z + a.tray_height +
                                            a.approach_above]), 'p_put')]
        best, why = None, []
        # first pose: every distinct IK solution (elbow / wrist branch) is tried, the one
        # closest to the current joints first: a branch can be fine over the handle and
        # still flip a joint later (carry, put down)
        p_above = steps[0][1]
        self.plan_grip = self.leg_grip('above')
        cands = {k: self.ik_above_all(p_above, math.remainder(yaw + k * math.pi, 2 * math.pi),
                                      q_now) for k in (0, 1)}
        carries = ([None] if place_xy is None else
                   [carry0 + i * a.carry_step for i in range(a.carry_tries)])
        variants = [(k, b, c) for k in (0, 1) for b in range(max(1, len(cands[k])))
                    for c in carries]
        done_k = set()
        for k, b, carry in variants:
            if k in done_k:                  # this yaw already planned (lowest carry wins)
                continue
            ty = math.remainder(yaw + k * math.pi, 2 * math.pi)
            legs, cost = {}, 0.0
            ty2 = ty if not turn else ty + turn             # tool yaw at the place
            # carry legs: high -> [via...] -> p_high, the tool yaw turned evenly over them
            carry_legs = [] if carry is None else self.carry_path(x, y, u, v, carry)
            yaw_of = {}
            for i, (cn, _) in enumerate(carry_legs):
                yaw_of[cn] = (ty + (ty2 - ty) * i / len(carry_legs),
                              ty + (ty2 - ty) * (i + 1) / len(carry_legs) if turn else None)
            all_steps = steps if carry is None else steps + [
                ('high', np.array([x, y, carry]), 'above')] + [
                (cn, cp, carry_legs[i - 1][0] if i else 'high')
                for i, (cn, cp) in enumerate(carry_legs)] + p_steps
            for name, p, frm in all_steps:
                self.plan_grip = self.leg_grip(name)
                yl = ty2 if name in ('p_put', 'p_lift') else ty
                y_end = None
                if name in yaw_of:
                    yl, y_end = yaw_of[name]
                if frm is None:
                    q, bad = (cands[k][b] if cands[k] else None), None
                    if q is None:
                        bad = (f'no IK at ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, '
                               f'{p[2] * 1000:.0f}) mm: {self.ik_why(p, ty, q_now)}')
                    q_prev = q_now
                else:
                    q_prev = legs[frm][1]
                    if name == 'p_lift':      # as high as reachable, at least 12 mm over
                        q, bad, p = self.lift_out(legs[frm][0], p, yl, q_prev)
                    else:
                        q, bad = self.line(legs[frm][0], p, yl, q_prev, y_end)
                if q is None:
                    why.append((ty, b, len(cands[k]), carry, name, bad))
                    legs = None
                    break
                cost += math.degrees(float(np.max(np.abs(q - q_prev))))
                legs[name] = (p, q)
            if legs is not None and 'p_lift' in legs and a.place_retreat > 0:
                # open fingers out of the way: straight toward the base (reach grows there)
                pl, ql = legs['p_lift']
                rr = float(np.linalg.norm(pl[:2]))
                if rr > 1e-6:
                    back = pl[:2] * max(0.0, 1.0 - a.place_retreat / rr)
                    whyb = []
                    self.plan_grip = self.leg_grip('p_back')
                    for dz in (0.03, 0.015, 0.0):
                        pb = np.array([back[0], back[1], pl[2] + dz])
                        # finer check: the arm is nearly stretched here, joints move fast
                        keep_sc, a.step_check = a.step_check, min(a.step_check, 0.003)
                        try:
                            qb, wb = self.line(pl, pb, ty2, ql)
                        finally:
                            a.step_check = keep_sc
                        if qb is not None:
                            legs['p_back'] = (pb, qb)
                            break
                        whyb.append(f'+{dz * 1000:.0f} mm: {wb}')
                    if 'p_back' not in legs:
                        print('  plan: no straight retreat toward the base after releasing ('
                              + '; '.join(whyb) + ')', flush=True)
            if legs is not None:
                done_k.add(k)
                if best is None or cost < best[0]:
                    best = (cost, ty, legs)
        if best is None:
            what = 'pick' + ('' if place_xy is None else ' + place')
            head = (f'{what} plan failed, nothing moved (handle ({x * 1000:.0f}, '
                    f'{y * 1000:.0f}) mm, r {math.hypot(x, y) * 1000:.0f} mm'
                    + ('' if place_xy is None else
                       f', place ({place_xy[0] * 1000:.0f}, {place_xy[1] * 1000:.0f}) mm, '
                       f'r {math.hypot(*place_xy[:2]) * 1000:.0f} mm') + ')')
            place_chk = None
            if place_xy is not None:
                place_chk = []
                for k in (0, 1):
                    ty = math.remainder(yaw + k * math.pi, 2 * math.pi)
                    ty2 = ty if not turn else ty + turn
                    pp = p_steps[0][1]
                    ok = self.ik_fixed(pp, ty2, q_now) is not None
                    place_chk.append(f'tool yaw {math.degrees(math.remainder(ty2, 2 * math.pi)):.0f}'
                                     + (': IK OK' if ok else
                                        ': ' + self.ik_why(pp, ty2, q_now)))
            raise Stop(head + '\n' + self.fail_report(why, place_chk))
        _, ty, legs = best
        if turn:
            print(f'  carry turns the tray {math.degrees(math.remainder(turn, 2 * math.pi)):+.0f}'
                  ' deg', flush=True)
        for name, (p, q) in legs.items():
            print(f'  plan {name:10s} tip ({p[0] * 1000:4.0f}, {p[1] * 1000:4.0f}, '
                  f'{p[2] * 1000:4.0f}) mm  J1..J6 [deg]: '
                  + ' '.join(f'{v:7.1f}' for v in np.degrees(q)), flush=True)
        return dict(tcp_yaw=ty, legs=legs,
                    carry=[n for n in legs if n.startswith('via')] + (['p_high']
                                                                       if 'p_high' in legs
                                                                       else []))

    @staticmethod
    def leg_stage(name):
        """Leg name -> 'pick' / 'carry' / 'place'."""
        if name in ('above', 'approach', 'grasp', 'put'):
            return 'pick'
        if name in ('p_put', 'p_lift'):
            return 'place'
        return 'carry'                   # high, via*, p_high

    def fail_report(self, why, place_chk=None):
        """_plan_pick failures [(ty, branch, n_branches, carry, leg, reason)] -> lines:
        a verdict (which side, pick / carry / place, stopped every variant; the stage
        reached furthest), then one line per distinct (tool yaw, branch, leg, reason) with
        the carry heights it held for collapsed into a range."""
        order = ('pick', 'carry', 'place')
        stages = {self.leg_stage(w[4]) for w in why}
        far = max(stages, key=order.index) if stages else 'pick'
        legs = sorted({w[4] for w in why if self.leg_stage(w[4]) == far})
        if far == 'pick':
            verdict = (f'PICK side not reachable ({", ".join(legs)} leg): every variant '
                       'fails before lifting the tray')
        elif far == 'carry':
            verdict = (f'CARRY not reachable ({", ".join(legs)} leg): the pick works, the '
                       'move across to the place does not')
        else:
            verdict = (f'PLACE side not reachable ({", ".join(legs)} leg): pick and carry '
                       'work, putting it down there does not')
        out = [f'  => {verdict}']
        if far == 'pick' and place_chk is not None:
            out.append('     place put-down point alone (not reached by the plan): '
                       + '; '.join(place_chk))
        groups = {}
        for ty, b, nb, carry, name, bad in why:
            key = (round(math.degrees(ty)), b, nb, name, bad)
            groups.setdefault(key, []).append(carry)
        for (tyd, b, nb, name, bad), cs in groups.items():
            cs = [c for c in cs if c is not None]
            ctxt = ''
            if cs:
                lo, hi = min(cs) * 1000, max(cs) * 1000
                ctxt = (f', carry z {lo:.0f} mm' if len(cs) == 1 else
                        f', carry z {lo:.0f}..{hi:.0f} mm ({len(cs)} tried)')
            out.append(f'     [{self.leg_stage(name)}] tool yaw {tyd}'
                       + (f', IK branch {b + 1}/{nb}' if nb > 1 else '')
                       + f'{ctxt}: {name} leg: {bad}')
        return '\n'.join(out)

    def lift_out(self, p0, p_nom, yaw, q0):
        """After releasing: straight up from p0 to p_nom (--approach-above over the handle
        top), or lower when that is out of reach, down to 12 mm over the handle top.
        -> (q, None, p) or (None, why, p_nom)."""
        a = self.args
        top = float(p_nom[2]) - a.approach_above
        first = None
        for c in [a.approach_above] + [c for c in (0.03, 0.02, 0.012) if c < a.approach_above]:
            p = np.array([p_nom[0], p_nom[1], top + c])
            q, bad = self.line(p0, p, yaw, q0)
            if q is not None:
                if c < a.approach_above:
                    print(f'  plan: after releasing only {c * 1000:.0f} mm up over the handle '
                          '(higher out of reach), then back toward the base', flush=True)
                return q, None, p
            first = first or bad
        return None, first, p_nom

    def carry_path(self, x, y, u, v, z):
        """Carry from above the pick (x, y) to above the place (u, v) at height z:
        -> [(name, point)] ending with 'p_high'. A straight line that passes closer than
        --carry-min-r to the base axis (the wrist flips there) is replaced by an arc around
        the base: via points every --carry-arc-step deg, radius never below --carry-min-r."""
        a = self.args
        p0, p1 = np.array([x, y]), np.array([u, v])
        d = p1 - p0
        t = 0.0 if d @ d < 1e-12 else float(np.clip(-(p0 @ d) / (d @ d), 0.0, 1.0))
        closest = float(np.linalg.norm(p0 + d * t))
        if closest >= a.carry_min_r or a.carry_min_r <= 0:
            return [('p_high', np.array([u, v, z]))]
        th0, th1 = math.atan2(y, x), math.atan2(v, u)
        dth = math.remainder(th1 - th0, 2 * math.pi)
        r0, r1 = float(np.linalg.norm(p0)), float(np.linalg.norm(p1))
        n = max(2, int(math.ceil(abs(math.degrees(dth)) / a.carry_arc_step)))
        out = []
        for i in range(1, n):
            th = th0 + dth * i / n
            r = max(a.carry_min_r, r0 + (r1 - r0) * i / n)
            out.append((f'via{i}', np.array([r * math.cos(th), r * math.sin(th), z])))
        memo = getattr(self, '_memo', None)
        if memo is None or 'arc_said' not in memo:
            print(f'  carry: straight line would pass {closest * 1000:.0f} mm from the base '
                  f'axis (< --carry-min-r): arc with {n - 1} via points', flush=True)
            if memo is not None:
                memo['arc_said'] = True
        return out + [('p_high', np.array([u, v, z]))]

    def ik_above_all(self, p, yaw, q_now):
        """First pose of the pick: IK from the current joints and from standard seeds
        facing the handle -> the distinct solutions (any joint > 10 deg apart), J1 closest
        to the handle direction (weighted) and the least motion first. Solutions with J1
        more than --max-j1-off from the handle direction (arm reaching over its back) are
        not used, except the reversed arm (J1 ~180 deg off, J2..J4 mirrored) as a fallback
        when no normal one is found (--no-flip = never)."""
        j1 = math.atan2(p[1], p[0])
        sols = self._ik_above(p, yaw, q_now, j1, flip=False)
        if not sols and not self.args.no_flip:
            sols = self._ik_above(p, yaw, q_now, j1, flip=True)
            if sols:
                print(f'  above ({p[0] * 1000:.0f}, {p[1] * 1000:.0f}) mm: no normal arm pose, '
                      f'reversed arm (J1 {math.degrees(sols[0][1][0]):+.0f} deg)', flush=True)
        return [q for _, q in sorted(sols, key=lambda t: t[0])]

    def _ik_above(self, p, yaw, q_now, j1, flip):
        """(cost, q) of the distinct solutions with J1 within --max-j1-off of the handle
        direction (flip: of the opposite direction, seeds mirrored)."""
        jc = math.remainder(j1 + (math.pi if flip else 0.0), 2 * math.pi)
        m = -1.0 if flip else 1.0
        seeds = ([] if flip else [q_now]) + [
            np.array([jc, m * j2, m * j3, 0.0, 0.0, 0.0]) for j2, j3 in SEEDS_J23] + [
            np.array([jc, m * a2, m * a3, m * a4, 0.0, 0.0]) for a2, a3, a4 in
            ((-0.8, -1.2, 0.4), (0.3, -1.8, 0.0), (-1.2, -0.6, 0.3), (0.6, -2.0, -0.2),
             (-1.2, 0.4, -0.7))]
        sols = []
        T = make_T(rot_z(yaw) @ DOWN, p)
        for sd in seeds:
            self._check()
            q = self.mv.compute_ik(T, TCP, ARM_JOINTS, sd, timeout=0.1,
                                   extra=self.grip_state())
            if q is None:
                continue
            q = np.asarray(q)
            off = abs(math.remainder(q[0] - jc, 2 * math.pi))
            if math.degrees(off) > self.args.max_j1_off:
                continue
            if any(np.max(np.abs(q - s)) < math.radians(10) for _, s in sols):
                continue
            sols.append((10.0 * off ** 2 + float(np.sum((q - q_now) ** 2)), q))
        return sols
