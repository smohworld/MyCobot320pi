"""Hardware backends for the myCobot 320 driver.

All backends speak degrees for the arm and 0..100 for the force-control (pro) gripper,
exactly like pymycobot. Only the driver's hardware thread calls these methods, so the
backends do not need to be thread-safe on their own.

  FakeRobot          no hardware; moves towards the last setpoint at a fixed speed
  PyMyCobotRobot     wraps pymycobot.MyCobot320Socket (Pi server, TCP 9000)
                     or pymycobot.MyCobot320 (USB serial)
"""
import time

# pymycobot RobotLimit for MyCobot320 / MyCobot320Socket (degrees)
ANGLE_MIN = [-168.0, -135.0, -145.0, -145.0, -168.0, -180.0]
ANGLE_MAX = [168.0, 135.0, 145.0, 145.0, 168.0, 180.0]


def clamp_angles(angles):
    return [min(max(a, lo), hi) for a, lo, hi in zip(angles, ANGLE_MIN, ANGLE_MAX)]


class RobotBackend:
    """Interface used by the driver node."""

    def connect(self):
        raise NotImplementedError

    def close(self):
        pass

    def get_angles(self):
        """Return 6 joint angles [deg] or None on a read error."""
        raise NotImplementedError

    def send_angles(self, angles_deg, speed):
        raise NotImplementedError

    def stop(self):
        pass

    def get_gripper(self):
        """Return gripper opening 0 (closed) .. 100 (open), or None."""
        return None

    def set_gripper(self, value, speed):
        pass

    def get_gripper_status(self):
        """0 moving, 1 stopped (no object), 2 object grasped, 3 object dropped; None unknown."""
        return None

    def set_gripper_torque(self, value):
        """Grip force setting of the force-control gripper (100..300), True if it was set."""
        return False

    def get_gripper_torque(self):
        """Grip force setting (100..300), or None."""
        return None

    def power_on(self):
        pass

    def release_servos(self):
        pass

    def focus_servos(self):
        pass


class FakeRobot(RobotBackend):
    """Kinematic simulation: joints move to the setpoint at speed% of max_vel_deg."""

    def __init__(self, max_vel_deg=120.0, gripper_vel=100.0, stop_short_deg=None, sag_deg=None):
        self._max_vel = max_vel_deg
        self._gripper_vel = gripper_vel
        # real servos stop a few deg before the target (in the direction of motion): set this
        # per joint to test the driver's settle correction without hardware
        self._short = list(stop_short_deg) if stop_short_deg else [0.0] * 6
        # and end this far below the goal regardless of direction (gravity sag of J2-J4)
        self._sag = list(sag_deg) if sag_deg else [0.0] * 6
        self._q = [0.0] * 6
        self._target = [0.0] * 6
        self._goal = [0.0] * 6        # last commanded angles
        self._dir = [0.0] * 6         # per joint direction of the commands (+-1, 0 = none yet)
        self._vel = max_vel_deg
        self._g = 100.0
        self._g_target = 100.0
        self._released = False
        self._t = time.monotonic()

    def connect(self):
        self._t = time.monotonic()

    def _step(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        if self._released:
            return
        step = self._vel * dt
        self._q = [q + max(-step, min(step, t - q)) for q, t in zip(self._q, self._target)]
        gstep = self._gripper_vel * dt
        self._g += max(-gstep, min(gstep, self._g_target - self._g))

    def get_angles(self):
        self._step()
        return list(self._q)

    def send_angles(self, angles_deg, speed):
        self._step()
        goal = clamp_angles(angles_deg)
        # stop `short` deg before a new goal on the side the commands come from (direction
        # of the previous goal -> this one; kept while the goal does not change) and `sag`
        # below it in either direction
        self._dir = [1.0 if g > pg + 1e-6 else -1.0 if g < pg - 1e-6 else d
                     for g, pg, d in zip(goal, self._goal, self._dir)]
        self._goal = list(goal)
        self._target = [g - s * d - a if abs(g - q) > s else q
                        for g, q, s, a, d in zip(goal, self._q, self._short, self._sag, self._dir)]
        self._vel = self._max_vel * max(1, min(100, speed)) / 100.0
        self._released = False

    def stop(self):
        self._step()
        self._target = list(self._q)

    def get_gripper(self):
        self._step()
        return int(round(self._g))

    def set_gripper(self, value, speed):
        self._step()
        self._g_target = float(value)

    def get_gripper_status(self):
        self._step()
        return 0 if abs(self._g - self._g_target) > 0.5 else 1

    def release_servos(self):
        self._released = True

    def focus_servos(self):
        self._released = False
        self._target = list(self._q)


class PyMyCobotRobot(RobotBackend):
    """myCobot 320 over socket (Pi server) or serial, via pymycobot."""

    def __init__(self, connection, ip='192.168.0.10', netport=9000,
                 serial_port='/dev/ttyUSB0', baudrate=115200,
                 gripper_id=14, use_gripper=True, logger=None):
        self.connection = connection
        self.ip, self.netport = ip, netport
        self.serial_port, self.baudrate = serial_port, baudrate
        self.gripper_id = gripper_id
        self.use_gripper = use_gripper
        self.log = logger
        self.mc = None
        self._gripper_speed = None
        self.drained_bytes = 0

    def connect(self):
        if self.connection == 'socket':
            from pymycobot import MyCobot320Socket
            self.mc = MyCobot320Socket(self.ip, self.netport)
        elif self.connection == 'serial':
            from pymycobot import MyCobot320
            self.mc = MyCobot320(self.serial_port, str(self.baudrate))
        else:
            raise ValueError(f'unknown connection {self.connection!r}')
        time.sleep(0.5)
        self._gripper_speed = None
        # 1 = always execute the newest command (needed for streaming setpoints)
        self.mc.set_fresh_mode(1)
        if self.mc.is_power_on() != 1:
            self.mc.power_on()
            time.sleep(1.0)

    def close(self):
        if self.mc is not None:
            try:
                self.mc.close()
            except Exception:  # noqa: BLE001
                pass

    def _drain(self):
        """Discard bytes left in the input buffer before a request.

        pymycobot reads ONE recv() per request and matches it by command id. A late reply
        (timeout, gripper not answering, a second client on the Pi server) stays in the
        buffer and is read as the answer to the NEXT request; the id check then fails, the
        call returns -1 and every later read is shifted by one -> get_angles keeps failing
        while send_angles still moves the arm."""
        n = 0
        try:
            if self.connection == 'socket':
                s = self.mc.sock
                s.setblocking(False)
                try:
                    while True:
                        d = s.recv(4096)
                        if not d:
                            break
                        n += len(d)
                except (BlockingIOError, InterruptedError):
                    pass
                finally:
                    s.setblocking(True)
            else:
                sp = self.mc._serial_port
                n = sp.in_waiting
                if n:
                    sp.reset_input_buffer()
        except (AttributeError, OSError):
            return
        if n:
            self.drained_bytes += n
            if self.log is not None:
                self.log.warn(f'discarded {n} stale bytes from the robot connection '
                              '(late reply / reply desync)', throttle_duration_sec=2.0)

    def get_angles(self):
        self._drain()
        a = self.mc.get_angles()
        if not isinstance(a, list) or len(a) != 6:
            return None
        return [float(x) for x in a]

    def send_angles(self, angles_deg, speed):
        angles = [round(a, 2) for a in clamp_angles(angles_deg)]
        self._drain()
        self.mc.send_angles(angles, int(max(1, min(100, speed))))

    def stop(self):
        self._drain()
        self.mc.stop()

    def get_gripper(self):
        if not self.use_gripper:
            return None
        self._drain()
        v = self.mc.get_pro_gripper_angle(self.gripper_id)
        if not isinstance(v, int) or v < 0 or v > 100:
            return None
        return v

    def set_gripper(self, value, speed):
        if not self.use_gripper:
            return
        speed = int(max(1, min(100, speed)))
        if speed != self._gripper_speed:
            self.mc.set_pro_gripper_speed(speed, self.gripper_id)
            self._gripper_speed = speed
        self.mc.set_pro_gripper_angle(int(max(0, min(100, round(value)))), self.gripper_id)

    def set_gripper_torque(self, value):
        if not self.use_gripper:
            return False
        return self.mc.set_pro_gripper_torque(int(value), self.gripper_id) == 1

    def get_gripper_torque(self):
        if not self.use_gripper:
            return None
        self._drain()
        v = self.mc.get_pro_gripper_torque(self.gripper_id)
        return v if isinstance(v, int) and 0 <= v <= 100 else None

    def get_gripper_status(self):
        if not self.use_gripper:
            return None
        s = self.mc.get_pro_gripper_status(self.gripper_id)
        return s if isinstance(s, int) and 0 <= s <= 3 else None

    def power_on(self):
        self.mc.power_on()

    def release_servos(self):
        self.mc.release_all_servos()

    def focus_servos(self):
        self.mc.focus_all_servos()
