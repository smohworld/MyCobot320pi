"""Thin wrapper around pymycobot for myCobot 320 + myGripper F100 (pro force gripper).

The same interface works over a TCP socket (PC -> server on the robot's Pi),
a USB/serial port, or a fake in-process robot for testing without hardware.
All angles here are in degrees (pymycobot units); the ROS node converts to rad.
"""
import threading
import time


def _valid_angles(a):
    return isinstance(a, (list, tuple)) and len(a) == 6 and all(
        isinstance(v, (int, float)) for v in a)


class MyCobotInterface:
    """Common API used by the driver node."""

    def __init__(self, connection, ip='192.168.0.10', netport=9000,
                 serial_port='/dev/ttyUSB0', baudrate=115200, gripper_id=14):
        self.connection = connection
        self.gripper_id = gripper_id
        # pymycobot already locks each command; this lock keeps multi-step
        # sequences (e.g. stop + read) from interleaving with the state poller.
        self.lock = threading.Lock()
        if connection == 'socket':
            from pymycobot import MyCobot320Socket
            self.mc = MyCobot320Socket(ip, netport)
        elif connection == 'serial':
            from pymycobot import MyCobot320
            self.mc = MyCobot320(serial_port, str(baudrate))
        elif connection == 'fake':
            self.mc = FakeMyCobot320()
        else:
            raise ValueError(f"connection must be 'socket', 'serial' or 'fake', got {connection!r}")

    # --- arm ---------------------------------------------------------------
    def get_angles(self):
        """Return 6 joint angles [deg] or None on a bad/empty reply."""
        with self.lock:
            a = self.mc.get_angles()
        return list(a) if _valid_angles(a) else None

    def send_angles(self, angles_deg, speed):
        with self.lock:
            return self.mc.send_angles([float(v) for v in angles_deg], int(speed))

    def is_moving(self):
        with self.lock:
            return self.mc.is_moving()

    def stop(self):
        with self.lock:
            return self.mc.stop()

    def power_on(self):
        with self.lock:
            return self.mc.power_on()

    def release_all_servos(self):
        with self.lock:
            return self.mc.release_all_servos()

    def focus_all_servos(self):
        with self.lock:
            return self.mc.focus_all_servos()

    # --- myGripper F100 (pro force-control gripper), 0 = closed .. 100 = open
    def get_gripper(self):
        with self.lock:
            v = self.mc.get_pro_gripper_angle(self.gripper_id)
        return float(v) if isinstance(v, (int, float)) and 0 <= v <= 100 else None

    def set_gripper(self, value):
        value = int(round(min(100.0, max(0.0, value))))
        with self.lock:
            return self.mc.set_pro_gripper_angle(value, self.gripper_id)

    def set_gripper_torque(self, torque):
        with self.lock:
            return self.mc.set_pro_gripper_torque(int(torque), self.gripper_id)

    def close(self):
        try:
            if self.connection == 'socket':
                self.mc.sock.close()
            elif self.connection == 'serial':
                self.mc.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass


class FakeMyCobot320:
    """Minimal simulated robot: joints move toward the target at a speed-dependent rate."""

    MAX_DEG_PER_S = 60.0  # at speed 100

    def __init__(self):
        self._angles = [0.0] * 6
        self._target = [0.0] * 6
        self._speed = 50
        self._gripper = 100.0
        self._t = time.monotonic()
        self._lock = threading.Lock()

    def _update(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        step = self.MAX_DEG_PER_S * self._speed / 100.0 * dt
        self._angles = [a + max(-step, min(step, t - a))
                        for a, t in zip(self._angles, self._target)]

    def get_angles(self):
        with self._lock:
            self._update()
            return [round(a, 2) for a in self._angles]

    def send_angles(self, angles, speed):
        with self._lock:
            self._update()
            self._target, self._speed = list(angles), max(1, min(100, speed))
        return 1

    def is_moving(self):
        with self._lock:
            self._update()
            return int(any(abs(a - t) > 0.05 for a, t in zip(self._angles, self._target)))

    def stop(self):
        with self._lock:
            self._update()
            self._target = list(self._angles)
        return 1

    def power_on(self):
        return 1

    def release_all_servos(self):
        return 1

    def focus_all_servos(self):
        return 1

    def get_pro_gripper_angle(self, gripper_id=14):
        return self._gripper

    def set_pro_gripper_angle(self, value, gripper_id=14):
        self._gripper = float(value)
        return 1

    def set_pro_gripper_torque(self, torque, gripper_id=14):
        return 1
