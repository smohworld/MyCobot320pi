"""Pick the tray by its grip pin, from above, keeping the tray level.

Needs (like red_cube_hover) move_group, the driver and the camera running (handeye.launch.py)
and the arm in a look-down pose (goto_lookdown / --search). The detection runs inside this node
(same code and options as tray_detect: YOLO + black-body rectangle, pin = tape centre); a
separate tray_detect is not needed.

Detection (d, and at the start of every p / v): --frames camera frames while the arm stands
still -> median pin xy + tray yaw (tray +x pointing away from the pin). When the tray touches
the image border or the gripper fingers (only partly seen) the camera moves to look straight
down at it (that pose becomes the new home) and detects again, at most --recenter times:
  pin (tape) visible: no move needed: pin xy from the tape; tray yaw = normal of the wall edge
                      next to the pin (line fit), else the tape edge axis; the rough direction
                      pin -> visible black part only picks which of those (side / axis); the
                      servo above the pin then corrects the xy
  no pin:             toward the visible black part, at most --look-step per move

Servo above the pin (p and v; --no-servo to skip): with the arm above the pin the camera
measures the tape again and the TCP is moved by  Kp * e + Ki * sum(e)  (e = pin - TCP in xy,
each move at most --servo-max-step) until |e| < --servo-tol or --servo-iters moves. Pin and TCP
are both seen through the same camera, so arm (joint) errors drop out; the hand-eye error stays.

Step mode (default): every motion is proposed in the window first and runs on Enter
(x / SPACE cancels): camera move, above the pin, descend, close, lift, and the place steps.
a toggles auto mode (no questions; --auto starts in it; without a window it is always auto).

  ros2 run mycobot_tasks tray_pick                       # window, plans only (dry run)
  ros2 run mycobot_tasks tray_pick --go --real           # window, moves the REAL robot
  ros2 run mycobot_tasks tray_pick --go --real --search --do pick

Window: left click on the desk = move the finger tips --click-height above that point;
        while holding the tray: put the tray down with its CENTRE at that point
        p pick   b put the tray back where it was picked   v move above the pin (approach height)
        h home   s search pose (new home)   d detect again   o / c open / close gripper
        u straight up   Enter next step   a auto / step mode   y live YOLO boxes on / off
        x / SPACE stop   q quit

Pick: open + above the pin at the highest of --servo-height (120 mm), 100, 85 mm, --approach
(70 mm) that has IK (finger tips above the desk; higher = the pin is further from the fingers
in the image) -> servo -> straight down in two legs (--approach, then --grasp-z = 17 mm: the
fingers cover the graspable part, 17..45 mm) -> close (the force gripper must report "object
grasped", else it opens again) -> lift straight up to --approach. The tool orientation is the
same on the way up, so the tray stays level.

Place on the desk (b, or a click while holding): the TCP keeps the yaw it had at the grasp
(the tray is not turned), so the pin goes to  centre - (--tray-len/2 + --pin-out/2) * tray x.
Above that pin point (approach height) -> straight down until the finger tips are --grasp-z +
--place-drop over the desk (the tray bottom then hangs --place-drop over it) -> open -> straight
up. The transfer between the two approach poses is a joint move: the tool ends level, but it can
tip a little on the way (fine for an empty tray, keep it short). Placing it on the TurtleBot is
not part of this node yet.

The fingers pinch the 20 mm wide pin. They close along the TCP x axis (URDF: finger joints at
gripper_base x = -44.5 / +44.2 mm), so the TCP yaw is the tray yaw + --grip-yaw-offset
(default 0; it was 90, which closes across the pin along the tray y axis). Check it in a dry
run (the plan prints the joints) and with the gripper open ABOVE the pin ('v') before the first grasp. The pin sticks out only 12 mm
of the tray wall: x error of the finger tips beyond ~3 mm drives them into the wall.
Without --go every action is only planned (IK) and printed.
"""
import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from mycobot_calibration.geometry import make_T, rot_z
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_tasks.red_cube_hover import (App, CAM, Cube, DOWN, Stop, TCP, _text, project,
                                                ray_plane)
from mycobot_tasks.red_cube_hover import LIMITS, SEEDS_J23, look_at
from mycobot_tasks.yolo_remote import CLASSES, make_yolo
from mycobot_tasks.tray_detect import (TrayDetector, add_detect_args,
                                       finish_detect_args, fingers_from_tf)


class TrayApp(App):
    def __init__(self, node, args):
        self.pin = None                   # dict(p=xyz of the pin, yaw) of the last detection
        self.grip = None                  # dict(pin=xy, tcp_yaw, tray_yaw) of the held tray
        self.tray = None                  # (centre xyz, yaw) of the last detection (window)
        self.prompt = None                # step waiting for Enter (window)
        self._go_on = threading.Event()
        self.auto = args.auto
        super().__init__(node, args)
        self.det = TrayDetector(args, self.K, self.D)

    # ------------------------------------------------------------ steps
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

    # ----------------------------------------------------------- detection
    def detect_cube(self):                # name kept: inherited actions (home, search) call it
        return self.detect_pin()

    def detect_once(self):
        """--frames fresh frames while the arm stands still -> (results, partial xys, last msg)."""
        a = self.args
        q0, t0 = self.node.q(), time.monotonic()
        res, partial, after = [], [], t0
        while len(res) + len(partial) < a.frames and time.monotonic() - t0 < a.pin_timeout:
            img = self.node.fresh(after, timeout=1.0)
            if img is None:
                continue
            after = time.monotonic()
            T = self.node.T('base', CAM)
            if T is None:
                raise Stop(f'no TF base -> {CAM}')
            r = self.det.process(img, T, ignore=fingers_from_tf(self.node.T, img.shape, self.K))
            if r is not None:
                res.append(r)
            elif self.det.partial is not None:
                partial.append(self.det.partial)
        q1 = self.node.q()
        if q0 is None or q1 is None or np.max(np.abs(q1 - q0)) > np.radians(0.5):
            raise Stop('the arm moved during detection: try again when it stands still')
        return res, partial, self.det.msg

    def detect_pin(self):
        """Detect (moving toward a partly seen tray) -> self.pin = dict(p, yaw)."""
        a = self.args
        self.pin = self.tray = None
        if a.go and not self.holding and self.node.js.get('gripper_controller', 1.0) < 0.5:
            self.say('opening the gripper first: open fingers stay out of the camera view')
            self.gripper(True)
            time.sleep(0.3)
        for attempt in range(a.recenter + 1):
            res, partial, msg = self.detect_once()
            if len(res) >= max(1, a.frames // 2):
                break
            if len(partial) < max(1, a.frames // 2) or attempt == a.recenter:
                if partial:
                    raise Stop(f'tray still only partly seen after {attempt} recentring moves')
                raise Stop(f'no tray / pin ({len(res)} of {a.frames} frames): {msg}')
            blob = np.median(np.array([p['blob'] for p in partial]), axis=0)
            with_pin = [p for p in partial if p['pin'] is not None]
            if len(with_pin) * 2 >= len(partial):
                return self.pin_from_partial(with_pin)
            else:
                self.say(f'tray partly seen (no pin): toward ({blob[0] * 1000:.0f}, '
                         f'{blob[1] * 1000:.0f}) mm ({attempt + 1}/{a.recenter})')
                self.look_at(blob)
            if not a.go:
                raise Stop('(dry run) the camera would move there first: run with --go')
        pins = np.array([r[2] for r in res])
        xy = np.median(pins, axis=0)
        yaws = np.array([r[1] for r in res])
        yaw = math.atan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))
        spread = float(np.max(np.linalg.norm(pins - xy, axis=1)))
        if spread > a.max_spread:
            raise Stop(f'pin pose not stable ({spread * 1000:.1f} mm spread > --max-spread '
                       f'{a.max_spread * 1000:.0f} mm): the tray or the arm is moving?')
        centre = np.median(np.array([r[0] for r in res]), axis=0)
        self.tray = (centre, yaw)
        self.pin = dict(p=np.array([xy[0], xy[1], a.table_z + a.pin_z]), yaw=yaw)
        self.say(f'pin ({xy[0] * 1000:.1f}, {xy[1] * 1000:.1f}) mm, tray yaw '
                 f'{math.degrees(yaw):.1f} deg, spread {spread * 1000:.1f} mm '
                 f'({len(res)} frames, tape-geo {self.det.pin_diff * 1000:.0f} mm)')
        return self.pin

    def pin_from_partial(self, parts):
        """Tray only partly seen but the pin is: pin xy from the tape, tray yaw (+x away from the
        pin) = direction pin -> visible black part, snapped to the nearest tape edge axis
        (mod 90 deg). No camera move: the servo above the pin corrects the xy."""
        a = self.args
        pins = np.array([p['pin'] for p in parts])
        xy = np.median(pins, axis=0)
        spread = float(np.max(np.linalg.norm(pins - xy, axis=1)))
        if spread > a.max_spread:
            raise Stop(f'pin pose not stable ({spread * 1000:.1f} mm spread > --max-spread '
                       f'{a.max_spread * 1000:.0f} mm): the tray or the arm is moving?')
        yaws, srcs, offs = [], [], []
        for p in parts:
            u = p['blob'] - p['pin']
            phi = math.atan2(u[1], u[0])            # rough tray +x: toward the visible part
            y, src = phi, 'black-part direction'
            # the black-part direction is only rough (a partly seen tray's centroid is biased
            # toward the visible side, 40 deg seen): it only picks the side / the axis
            for key, step, tol, name in (('wall', math.pi, 90.0, 'wall edge'),
                                         ('axis', math.pi / 2, 45.0, 'tape edge')):
                if p.get(key) is None:
                    continue
                # tray x is perpendicular to the wall; the tape edges are along x or y
                base = p[key] + (math.pi / 2 if key == 'wall' else 0.0)
                cand = [base + k * step for k in range(int(round(2 * math.pi / step)))]
                c = min(cand, key=lambda c: abs(math.remainder(c - phi, 2 * math.pi)))
                off = math.degrees(abs(math.remainder(c - phi, 2 * math.pi)))
                if off <= tol:
                    y, src = c, name
                    offs.append(off)
                    break
            yaws.append(y)
            srcs.append(src)
        yaws = np.array(yaws)
        yaw = math.atan2(np.median(np.sin(yaws)), np.median(np.cos(yaws)))
        d = a.tray_len / 2 + a.pin_out / 2
        centre = np.array([xy[0] + d * math.cos(yaw), xy[1] + d * math.sin(yaw),
                           a.table_z + a.tray_h])
        self.tray = (centre, yaw)
        self.pin = dict(p=np.array([xy[0], xy[1], a.table_z + a.pin_z]), yaw=yaw, partial=True)
        src = max(set(srcs), key=srcs.count)
        how = src + (f' ({np.median(offs):.0f} deg from the black-part direction)' if offs
                     else '')
        self.say(f'tray partly seen, pin ({xy[0] * 1000:.1f}, {xy[1] * 1000:.1f}) mm, tray yaw '
                 f'{math.degrees(yaw):.1f} deg from the {how}, spread {spread * 1000:.1f} mm')
        return self.pin

    def above_pose(self, x, y, yaw):
        """Highest pose straight above (x, y) from --servo-height down to --approach that has
        IK -> (z, q, tcp yaw used)."""
        a = self.args
        zs = sorted({round(h, 4) for h in (a.servo_height, 0.10, 0.085, a.approach)
                     if h >= a.approach - 1e-9}, reverse=True)
        seed = self.node.q()
        for h in zs:
            self._check()
            r = self.ik_down(np.array([x, y, a.table_z + h]), yaw, seed)
            if r is not None:
                return a.table_z + h, r[0], r[1]
        raise Stop(f'no IK above ({x * 1000:.0f}, {y * 1000:.0f}) mm at any of '
                   f'{[int(h * 1000) for h in zs]} mm (out of reach / J6 window)')

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

    # ------------------------------------------------------------- servo
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

    def measure_pin(self, n=None):
        """Median tape xy over n (--frames) fresh frames (arm standing still) or None."""
        a = self.args
        n = n or a.frames
        xs, after, t0 = [], time.monotonic(), time.monotonic()
        while len(xs) < n and time.monotonic() - t0 < a.pin_timeout:
            img = self.node.fresh(after, timeout=1.0)
            if img is None:
                continue
            after = time.monotonic()
            T = self.node.T('base', CAM)
            if T is not None:
                p = self.det.pin_only(img, T)
                if p is not None:
                    xs.append(p)
        return np.median(np.array(xs), axis=0) if len(xs) > n // 2 else None

    def servo(self):
        """PI loop in xy at the current height and tool yaw until the TCP is over the pin.
        Returns the pin xy (last measurement)."""
        a = self.args
        yaw = self.tcp_yaw()
        integ = np.zeros(2)
        pin = None
        t_start = time.monotonic()
        for i in range(a.servo_iters + 1):
            t0 = time.monotonic()
            self.wait_still()
            t1 = time.monotonic()
            pin = self.measure_pin(a.servo_frames)
            t2 = time.monotonic()
            T = self.node.T('base', TCP)
            if pin is None or T is None:
                raise Stop('servo: pin not visible from here (try a higher --approach, e.g. 0.10)')
            tcp = T[:3, 3]
            e = pin - tcp[:2]
            en = float(np.linalg.norm(e))
            self.say(f'servo {i}: error {en * 1000:.1f} mm (dx {e[0] * 1000:+.1f}, '
                     f'dy {e[1] * 1000:+.1f})  [settle {t1 - t0:.2f} s, measure {t2 - t1:.2f} s]')
            if en < a.servo_tol:
                self.say(f'servo: centred over the pin ({en * 1000:.1f} mm, '
                         f'{time.monotonic() - t_start:.1f} s)')
                return pin
            if i == a.servo_iters:
                break
            if en > a.servo_max_err:
                raise Stop(f'servo: error {en * 1000:.0f} mm > --servo-max-err (wrong pin box?)')
            integ += e
            step = a.servo_kp * e + a.servo_ki * integ
            sn = float(np.linalg.norm(step))
            if sn > a.servo_max_step:
                step *= a.servo_max_step / sn
            tgt = np.array([tcp[0] + step[0], tcp[1] + step[1], tcp[2]])
            q = self.ik_fixed(tgt, yaw, self.node.q())
            if q is None:
                raise Stop('servo: no IK for the correction')
            self.target = tgt
            t3 = time.monotonic()
            self.move(f'servo {i + 1}', q, velocity=a.servo_velocity)
            print(f'  servo move {i + 1}: {time.monotonic() - t3:.2f} s', flush=True)
        self.say(f'servo: not below --servo-tol after {a.servo_iters} moves '
                 f'({en * 1000:.1f} mm left)')
        return pin

    # ---------------------------------------------------------------- IK
    def ik_down(self, p, yaw, seed):
        """TCP at p pointing down with the yaw (or yaw + 180 deg: same pinch, least motion).
        Not yaw + 90: the fingers must keep the same pinch direction."""
        best = None
        for k in (0, 1):
            y = yaw + k * math.pi
            q = self.mv.compute_ik(make_T(rot_z(y) @ DOWN, p), TCP, ARM_JOINTS, seed,
                                   timeout=0.1)
            if q is not None:
                cost = float(np.max(np.abs(np.asarray(q) - seed)))
                if best is None or cost < best[0]:
                    best = (cost, np.asarray(q), y)
        return None if best is None else (best[1], best[2])

    # ----------------------------------------------------------- actions
    def grip_points(self):
        """Detect -> (pin xy, tool yaw wanted)."""
        a = self.args
        if a.grasp_z <= a.table_margin:
            raise Stop(f'--grasp-z {a.grasp_z} must be above --table-margin {a.table_margin}')
        pin = self.detect_pin()
        return pin['p'][:2].copy(), pin['yaw'] + math.radians(a.grip_yaw_offset)

    def go_above(self, xy, yaw, open_gripper=False):
        """Move above the pin at the highest reachable height, servo there -> (xy, tcp yaw)."""
        a = self.args
        z, q, tcp_yaw = self.above_pose(xy[0], xy[1], yaw)
        above = np.array([xy[0], xy[1], z])
        print('  above pin  J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
        self.target = above
        self.confirm(('open + ' if open_gripper else '') + f'above the pin ({xy[0] * 1000:.0f}, '
                     f'{xy[1] * 1000:.0f}) mm, finger tips {(z - a.table_z) * 1000:.0f} mm '
                     'over the desk')
        opened = self.gripper_async(True) if open_gripper else (lambda: None)
        self.move('above pin', q)
        opened()
        self.tip_report(above)
        if a.servo and a.go:
            self.confirm('servo: centre the TCP over the pin with the camera')
            self.servo()
            T = self.node.T('base', TCP)
            xy, tcp_yaw = T[:2, 3].copy(), self.tcp_yaw()
        return xy, tcp_yaw

    def act_hover(self):
        xy, yaw = self.grip_points()
        self.go_above(xy, yaw, open_gripper=True)
        self.say('above the pin' + (' (servoed)' if self.args.servo and self.args.go else ''))

    def act_pick(self):
        a = self.args
        if self.holding:
            raise Stop('already holding the tray')
        xy, yaw = self.grip_points()
        pin_yaw = self.pin['yaw']
        xy, tcp_yaw = self.go_above(xy, yaw, open_gripper=True)
        seed = self.node.q() if a.go else None
        if seed is None:
            seed = self.above_pose(xy[0], xy[1], yaw)[1]
        app = np.array([xy[0], xy[1], a.table_z + a.approach])
        grasp = np.array([xy[0], xy[1], a.table_z + a.grasp_z])
        q_app = self.ik_fixed(app, tcp_yaw, seed)
        q_grasp = None if q_app is None else self.ik_fixed(grasp, tcp_yaw, q_app)
        if q_app is None or q_grasp is None:
            raise Stop('no IK for the descent below the pin')
        for n, q in (('approach', q_app), ('grasp', q_grasp)):
            print(f'  {n:10s} J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
        self.target = grasp
        self.confirm(f'descend: finger tips to {a.grasp_z * 1000:.0f} mm over the desk')
        self.move('down to approach', q_app, slow=True)      # two legs: stays near vertical
        self.move('grasp', q_grasp, slow=True, velocity=a.grasp_velocity)
        self.tip_report(grasp)
        self.confirm('close the gripper')
        r = self.gripper(False)
        missed = r is not None and not r.stalled
        if missed:
            self.gripper(True)
        else:
            self.holding = True
            self.grip = dict(pin=grasp[:2].copy(), tcp_yaw=tcp_yaw, tray_yaw=pin_yaw)
        self.confirm('lift straight up' + (' (MISSED: gripper open again)' if missed else ''))
        self.set_table(margin=False)
        try:
            self.move('lift', q_app, slow=True)     # same orientation: the tray stays level
        finally:
            self.set_table(margin=True)
        if missed:
            raise Stop('gripper closed without the pin (missed): it stays open; '
                       'd = detect again')
        self.pin = self.tray = None
        self.say('tray picked' + ('' if a.go else ' (dry run)') +
                 f': lifted level to {a.approach * 1000:.0f} mm (finger tips)')

    def ik_fixed(self, p, yaw, seed):
        """TCP at p pointing down with exactly this yaw (the held tray must not turn)."""
        q = self.mv.compute_ik(make_T(rot_z(yaw) @ DOWN, p), TCP, ARM_JOINTS, seed, timeout=0.1)
        return None if q is None else np.asarray(q)

    def act_place(self, centre=None):
        """Put the held tray down: its centre at `centre` (desk xy), None = where it was picked."""
        a = self.args
        if not self.holding and a.go:
            raise Stop('not holding the tray (p = pick)')
        g = self.grip
        if g is None:                       # dry run without a pick, or after --holding
            yaw = self.tcp_yaw()
            g = dict(pin=None, tcp_yaw=yaw, tray_yaw=yaw - math.radians(a.grip_yaw_offset))
        if centre is None:
            if g['pin'] is None:
                raise Stop('pick position unknown (restarted?): click the desk to place')
            pin = g['pin']
        else:
            d = a.tray_len / 2 + a.pin_out / 2
            pin = np.asarray(centre[:2]) - d * np.array([math.cos(g['tray_yaw']),
                                                         math.sin(g['tray_yaw'])])
        x, y = pin
        above = np.array([x, y, a.table_z + a.approach])
        put = np.array([x, y, a.table_z + a.grasp_z + a.place_drop])
        self.target = put
        seed = self.node.q()
        q_above = self.ik_fixed(above, g['tcp_yaw'], seed)
        q_put = None if q_above is None else self.ik_fixed(put, g['tcp_yaw'], q_above)
        if q_above is None or q_put is None:
            raise Stop(f'no IK for placing the pin at ({x * 1000:.0f}, {y * 1000:.0f}) mm with '
                       'the tray yaw kept (out of reach / J6 window)')
        for n, q in (('place above', q_above), ('put down', q_put)):
            print(f'  {n:10s} J1..J6 [deg]: ' + ' '.join(f'{v:7.1f}' for v in np.degrees(q)))
        self.confirm(f'carry: pin above ({x * 1000:.0f}, {y * 1000:.0f}) mm')
        self.move('place above', q_above)
        self.set_table(margin=False)          # closed fingers hang into the margin box
        try:
            self.confirm(f'put down: tray bottom {a.place_drop * 1000:.0f} mm over the desk')
            self.move('put down', q_put, slow=True, velocity=a.grasp_velocity)
            self.tip_report(put)
            self.confirm('open the gripper (release the tray)')
            self.gripper(True)
            self.holding, self.grip = False, None
            self.confirm('lift straight up')
            self.move('lift', q_above, slow=True)
        finally:
            self.set_table(margin=True)
        self.say('tray placed' + ('' if a.go else ' (dry run)') +
                 f': pin at ({x * 1000:.0f}, {y * 1000:.0f}) mm')

    # ------------------------------------------------------------ clicks
    def click(self, u, v, _on_cube=False):
        """Left click on the desk: move the finger tips --click-height above that point
        (keeps the current tool yaw). The tray is never placed by a click."""
        T_bc = self.node.T('base', CAM)
        if T_bc is None:
            self.say(f'no TF base -> {CAM}')
            return
        p = ray_plane([u, v], self.K, self.D, T_bc, self.args.table_z)[0]
        axis = T_bc[:3, 2]                         # optical axis in base
        tilt = math.degrees(math.acos(min(1.0, abs(axis[2]))))
        print(f'click px ({u}, {v}) -> desk point ({p[0] * 1000:.1f}, {p[1] * 1000:.1f}) mm '
              f'at z {self.args.table_z * 1000:.0f} mm | camera at ({T_bc[0, 3] * 1000:.0f}, '
              f'{T_bc[1, 3] * 1000:.0f}, {T_bc[2, 3] * 1000:.0f}) mm, optical axis '
              f'{tilt:.1f} deg from vertical', flush=True)
        if self.holding:
            self.submit('place', self.act_place, p)
        else:
            self.submit('move', self.act_move_to, p)


# ------------------------------------------------------------------ window
def view(app):
    node, args = app.node, app.args
    win = 'tray_pick'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(ev, u, v, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            app.click(u, v)

    cv2.setMouseCallback(win, on_mouse)
    keys = {ord('p'): ('pick', app.act_pick), ord('v'): ('above pin', app.act_hover),
            ord('b'): ('put back', app.act_place),
            ord('h'): ('home', app.act_home), ord('d'): ('detect', app.act_detect),
            ord('u'): ('up', app.act_up), ord('s'): ('search pose', app.act_search),
            ord('o'): ('open', lambda: app.gripper(True)),
            ord('c'): ('close', lambda: app.gripper(False))}
    # live YOLO overlay: its own net (cv2.dnn is not thread safe, the worker uses app.det)
    yolo = make_yolo(args.model, CLASSES, args.imgsz, args.conf, args.yolo_sock)
    show_yolo = True
    after = 0.0
    dets, t_yolo = [], 0.0
    while True:
        img = node.fresh(after, timeout=0.1)
        if img is not None:
            after = time.monotonic()
            vis = img.copy()
            # CPU inference: at most --live-hz (1 Hz while an action runs: the detection /
            # servo of the action need the CPU; every camera frame starved them)
            hz = args.live_hz if not app.busy else min(args.live_hz, 1.0)
            if show_yolo and hz > 0 and after - t_yolo >= 1.0 / hz:
                t_yolo = after
                try:
                    dets = yolo.detect_obb(img)
                except ValueError:                 # not an OBB model: axis-aligned boxes
                    dets = [(n, c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0,
                             np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
                            for n, c, x0, y0, x1, y1 in yolo.detect(img)]
            if show_yolo:
                for n, c, cx, cy, bw, bh, ang, corners in dets:
                    col = (0, 255, 0) if n == 'tray' else (0, 165, 255)
                    cv2.polylines(vis, [np.int32(corners)], True, col, 2)
                    cv2.circle(vis, (int(cx), int(cy)), 3, col, -1)
                    x0, y0 = np.int32(corners.min(axis=0))
                    _text(vis, f'{n} {c:.2f}', (int(x0), max(14, int(y0) - 5)), 0.45)
                _text(vis, f'YOLO {hz:.0f} Hz  (y = off)', (vis.shape[1] - 190, 25), 0.45)
            T_bc = node.T('base', 'camera_optical_frame')
            if T_bc is not None:
                for p, col in ((app.pin['p'] if app.pin else None, (0, 255, 255)),
                               (app.target, (255, 0, 255))):
                    if p is not None:
                        uv = project(p, T_bc, app.K, app.D)
                        if uv:
                            cv2.drawMarker(vis, uv, col, cv2.MARKER_TILTED_CROSS, 18, 2)
                if app.tray is not None:
                    c, yaw = app.tray
                    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                    pts = []
                    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                        xy = c[:2] + R @ [sx * args.tray_len / 2, sy * args.tray_wid / 2]
                        pts.append(project(np.array([xy[0], xy[1], c[2]]), T_bc, app.K, app.D))
                    if all(pts):
                        cv2.polylines(vis, [np.int32(pts)], True, (0, 0, 255), 2)
                if app.target is not None:
                    # X above is the finger-tip goal (in the air, so it shifts in the image);
                    # the cross below is the same x,y ON the desk, where you clicked.
                    foot = np.array([app.target[0], app.target[1], args.table_z])
                    uv = project(foot, T_bc, app.K, app.D)
                    if uv:
                        cv2.drawMarker(vis, uv, (255, 0, 255), cv2.MARKER_CROSS, 18, 1)
                        cv2.circle(vis, uv, 4, (255, 0, 255), -1)
            _text(vis, app.status, (10, 25), 0.55)
            _text(vis, f"{'' if args.go else 'DRY RUN  '}{'BUSY  ' if app.busy else ''}"
                       f"{'HOLDING  ' if app.holding else ''}{'AUTO' if app.auto else 'STEP'}",
                  (10, 50), 0.5)
            if app.prompt:
                _text(vis, f'NEXT: {app.prompt}', (10, 78), 0.55)
                _text(vis, 'Enter = go   x = cancel   a = auto', (10, 102), 0.55)
            _text(vis, ('click = place tray centre there  b put back  ' if app.holding else
                        'click = move above desk point  p pick  v above pin  ') +
                       'h home  s search  u up  o/c grip  d detect  a auto  x stop  q quit',
                  (10, vis.shape[0] - 12), 0.45)
            cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            break
        if k in (ord('x'), ord(' ')):
            app.stop()
        elif k in (13, 10, 141):                    # Enter (also keypad Enter)
            app._go_on.set()
        elif k == ord('y'):
            show_yolo = not show_yolo
        elif k == ord('a'):
            app.auto = not app.auto
            app.say('auto mode: no step questions' if app.auto else 'step mode: Enter per step')
            if app.auto:
                app._go_on.set()
        elif k in keys:
            app.submit(*keys[k])
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyWindow(win)


# -------------------------------------------------------------------- main
def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--approach', type=float, default=0.07,
                   help='finger tips above the desk for the approach / lift [m]')
    a.add_argument('--click-height', type=float, default=0.05,
                   help='finger tips above the desk for a clicked desk point [m]')
    a.add_argument('--grasp-z', type=float, default=0.017,
                   help='finger tips above the desk when grasping [m] (> --table-margin); '
                        '17 mm: the fingers cover the graspable part 17..45 mm')
    a.add_argument('--grip-yaw-offset', type=float, default=0.0,
                   help='TCP yaw relative to the tray x axis [deg]; the fingers close along '
                        'the TCP x axis, so 0 = they close along the tray x axis, 90 = along the tray y axis '
                        '(the former default). 0 and 180 pinch the same way')
    a.add_argument('--open-position', type=float, default=1.0,
                   help='gripper open command (joint value, 1.0 = fully open = 10 cm)')
    a.add_argument('--place-drop', type=float, default=0.003,
                   help='placing: the tray bottom stops this far above the desk, then opens [m]')
    a.add_argument('--table-margin', type=float, default=0.012,
                   help='desk collision box raised by this [m]')
    a.add_argument('--frames', type=int, default=5, help='camera frames per detection')
    a.add_argument('--pin-timeout', type=float, default=4.0, help='wait for them [s]')
    a.add_argument('--recenter', type=int, default=3,
                   help='tray partly seen: move the camera toward it at most this many times')
    a.add_argument('--servo-height', type=float, default=0.12,
                   help='go above the pin (and servo) at the highest of this, 100, 85 mm and '
                        '--approach that has IK (finger tips over the desk) [m]')
    a.add_argument('--look-step', type=float, default=0.08,
                   help='each such move shifts the viewed desk point at most this far [m]')
    a.add_argument('--look-max-r', type=float, default=0.30,
                   help='viewed desk point at most this far from the base [m]')
    a.add_argument('--max-spread', type=float, default=0.006,
                   help='refuse when the pin xy jumps more than this between messages [m]')
    add_detect_args(a)
    a.add_argument('--velocity', type=float, default=0.6,
                   help='velocity scaling for the free moves')
    a.add_argument('--slow-velocity', type=float, default=0.25,
                   help='velocity scaling near the desk: grasp descent and lift')
    a.add_argument('--grasp-velocity', type=float, default=None,
                   help='velocity scaling of the descent only (default: --slow-velocity)')
    a.add_argument('--accel', type=float, default=0.5, help='acceleration scaling')
    a.add_argument('--search', action='store_true',
                   help='go to the search pose first (it becomes home), then --do / detect')
    a.add_argument('--search-x', type=float, default=0.25)
    a.add_argument('--search-y', type=float, default=0.0)
    a.add_argument('--search-height', type=float, default=0.28)
    a.add_argument('--search-min-height', type=float, default=0.18)
    a.add_argument('--do', choices=['', 'none', 'detect', 'hover', 'pick'], default='',
                   help='run this at start (none = nothing, e.g. for click-to-move without tray_detect)')
    a.add_argument('--no-servo', dest='servo', action='store_false',
                   help='no camera correction above the pin')
    a.add_argument('--servo-kp', type=float, default=1.0, help='servo proportional gain')
    a.add_argument('--servo-frames', type=int, default=3, help='camera frames per servo measure')
    a.add_argument('--servo-velocity', type=float, default=0.5,
                   help='velocity scaling of the (small) servo moves')
    a.add_argument('--servo-ki', type=float, default=0.0, help='servo integral gain')
    a.add_argument('--servo-tol', type=float, default=0.002, help='servo done below this [m]')
    a.add_argument('--servo-iters', type=int, default=5, help='servo correction moves at most')
    a.add_argument('--servo-max-step', type=float, default=0.01, help='per move at most [m]')
    a.add_argument('--servo-max-err', type=float, default=0.03,
                   help='refuse corrections above this (wrong detection) [m]')
    a.add_argument('--go', action='store_true', help='really move (else plan only)')
    a.add_argument('--auto', action='store_true',
                   help='no step questions (default: each motion waits for Enter in the window)')
    a.add_argument('--real', action='store_true', help='allow moving a real (non-fake) robot')
    a.add_argument('--no-show', action='store_true', help='no camera window')
    a.add_argument('--live-hz', type=float, default=4.0,
                   help='live YOLO boxes in the window at most this often (0 = off)')
    a.add_argument('--cv-threads', type=int, default=2,
                   help='OpenCV threads (cv2.dnn on the CPU takes all cores otherwise)')
    a.add_argument('--holding', action='store_true',
                   help='start as holding the tray (after a restart)')
    args, _ = a.parse_known_args()             # ignore --ros-args from ros2 run
    cv2.setNumThreads(args.cv_threads)
    finish_detect_args(args)
    args.return_home = False
    show = not args.no_show
    if show and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        print('no display: running without the camera window')
        show = False
    if not show and not args.do:
        args.do = 'detect'
    if not show:
        args.auto = True                       # nobody can press Enter

    rclpy.init()
    node = Cube(args.image_topic)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        app = TrayApp(node, args)
        app.say('ready' + ('' if args.go else ' (DRY RUN: add --go to move)'))
        first = {'detect': ('detect', app.act_detect), 'hover': ('above pin', app.act_hover),
                 'pick': ('pick', app.act_pick)}
        if args.holding and not args.do:
            args.do = 'none'
        if args.search:
            name, fn = first.get(args.do, ('detect', None))
            app.submit('search pose', app.act_search, fn)
        elif args.do != 'none':
            app.submit(*first.get(args.do, ('detect', app.act_detect)))
        if show:
            view(app)
            if app.busy:
                print('window closed: waiting for the running action (Ctrl+C stops)')
        app.jobs.join()
    except KeyboardInterrupt:
        if 'app' in locals():
            app.stop()
    finally:
        cv2.destroyAllWindows()
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
