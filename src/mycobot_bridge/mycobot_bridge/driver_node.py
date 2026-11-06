"""myCobot 320 driver for MoveIt2 (moveit_simple_controller_manager, no ros2_control).

Interfaces
  pub  /joint_states                                  sensor_msgs/JointState (6 arm + gripper joint)
  act  /arm_controller/follow_joint_trajectory        control_msgs/FollowJointTrajectory
  act  /gripper_controller/gripper_cmd                control_msgs/GripperCommand
  srv  ~/power_on ~/release_servos ~/focus_servos ~/stop   std_srvs/Trigger

connection:=socket  -> pymycobot.MyCobot320Socket(ip, netport)   (Server_320.py on the Pi)
connection:=serial  -> pymycobot.MyCobot320(serial_port, baudrate)
connection:=fake    -> kinematic simulation, no hardware

Threading: one hardware thread owns the robot connection. It streams trajectory setpoints
(send_angles with fresh mode 1), reads joint angles / gripper state and publishes
/joint_states. Action and service callbacks only hand jobs to that thread and wait, so the
connection is never used from two threads at once.
"""
import math
import threading
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from mycobot_bridge.backends import ANGLE_MAX, ANGLE_MIN, FakeRobot, PyMyCobotRobot

ARM_JOINTS = [
    'joint2_to_joint1',
    'joint3_to_joint2',
    'joint4_to_joint3',
    'joint5_to_joint4',
    'joint6_to_joint5',
    'joint6output_to_joint6',
]
FJT = FollowJointTrajectory.Result


def _dur(d):
    return d.sec + d.nanosec * 1e-9


class TrajJob:
    """Trajectory in robot joint order, degrees, already time-scaled."""

    def __init__(self, times, positions):
        self.times = times            # (N,)  seconds from start
        self.positions = positions    # (N,6) degrees
        self.t_start = None           # monotonic, set by hw thread on first tick
        self.final_sent = False
        self.final_sent_t = 0.0
        self.last_sent = None
        self.desired = positions[0].copy()
        self.cancel = False
        self.preempted = False
        self.done = threading.Event()
        self.error_code = FJT.SUCCESSFUL
        self.error_string = ''
        # settle correction (after the trajectory end)
        self.cmd = None               # last commanded final angles (target + corrections)
        self.corrections = 0
        self.ref_q = None             # joints when they last moved (settle detection)
        self.ref_t = 0.0
        self.err0 = None              # error before the first correction [deg]
        self.bias = np.zeros(6)       # feedforward added to every command [deg]

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, t):
        t = min(max(t, 0.0), self.duration)
        return np.array([np.interp(t, self.times, self.positions[:, j]) for j in range(6)])

    def finish(self, code, msg=''):
        if not self.done.is_set():
            self.error_code, self.error_string = code, msg
            self.done.set()


class GripperJob:
    def __init__(self, target):
        self.target = target          # 0..100
        self.sent = False
        self.t_start = None
        self.cancel = False
        self.done = threading.Event()
        self.reached = False
        self.stalled = False
        self.ok = True
        self.msg = ''

    def finish(self, ok, reached=False, stalled=False, msg=''):
        if not self.done.is_set():
            self.ok, self.reached, self.stalled, self.msg = ok, reached, stalled, msg
            self.done.set()


class Request:
    """One-shot backend call executed in the hardware thread."""

    def __init__(self, fn, name):
        self.fn, self.name = fn, name
        self.done = threading.Event()
        self.ok, self.msg = False, ''


class MyCobotDriver(Node):

    def __init__(self):
        super().__init__('mycobot_driver')
        p = self.declare_parameter
        self.connection = p('connection', 'fake').value          # fake | socket | serial
        ip = p('ip', '192.168.0.10').value
        netport = p('netport', 9000).value
        serial_port = p('serial_port', '/dev/ttyUSB0').value
        baudrate = p('baudrate', 115200).value
        self.use_gripper = p('use_gripper', True).value
        gripper_id = p('gripper_id', 14).value

        self.loop_rate = p('loop_rate', 20.0).value              # hw loop [Hz]
        self.read_divider = max(1, p('read_divider_executing', 2).value)
        self.gripper_read_divider = max(1, p('gripper_read_divider', 10).value)
        self.stream_speed = p('stream_speed', 80).value          # send_angles speed while streaming
        self.lookahead = p('lookahead', 0.1).value               # [s] setpoint ahead of time
        self.min_step_deg = p('min_step_deg', 0.1).value         # skip setpoints smaller than this
        self.max_joint_vel = p('max_joint_vel_deg', 30.0).value  # [deg/s] stretch faster trajectories
        # Robot <-> URDF angle mapping: urdf_deg = sign * robot_deg + offset.
        # Identity unless the real arm and RViz disagree (see README / check procedure).
        self.signs = np.array(p('joint_signs', [1.0] * 6).value, dtype=float)
        self.offsets = np.array(p('joint_offsets_deg', [0.0] * 6).value, dtype=float)
        if self.signs.shape != (6,) or self.offsets.shape != (6,) or \
                not np.all(np.isin(self.signs, (-1.0, 1.0))):
            raise ValueError('joint_signs must be 6 values of +-1, joint_offsets_deg 6 values')
        # Software joint limits [deg, URDF frame]. J6 is +-90 so the camera/gripper cable
        # cannot wind up. Trajectories outside are rejected; every command is clamped to them.
        hw_a = self.signs * np.array(ANGLE_MIN) + self.offsets
        hw_b = self.signs * np.array(ANGLE_MAX) + self.offsets
        j_min = p('joint_min_deg', ANGLE_MIN[:5] + [-90.0]).value
        j_max = p('joint_max_deg', ANGLE_MAX[:5] + [90.0]).value
        self.j_min = np.maximum(np.array(j_min, dtype=float), np.minimum(hw_a, hw_b))
        self.j_max = np.minimum(np.array(j_max, dtype=float), np.maximum(hw_a, hw_b))
        self.limit_margin = p('limit_margin_deg', 1.0).value      # tolerance before rejecting
        self.start_tol = p('start_tolerance_deg', 5.0).value
        self.goal_tol = p('goal_tolerance_deg', 2.0).value
        self.goal_timeout = p('goal_timeout', 3.0).value         # [s] after trajectory end
        # Settle correction: the servos stop 2-3 deg short of the target. Once the arm has
        # settled, command target + remaining error (per joint, accumulated) so it ends up
        # on the target. max_corrections 0 disables it.
        self.settle_tol = p('settle_tolerance_deg', 0.5).value     # done when all within
        self.max_corrections = p('max_corrections', 3).value
        self.corr_deadband = p('correction_deadband_deg', 0.3).value  # leave smaller errors
        self.max_corr = p('max_correction_deg', 6.0).value         # |command - target| cap
        self.corr_speed = p('correction_speed', 30).value          # send_angles speed
        self.settle_time = p('settle_time', 0.4).value             # [s] no motion = settled
        self.settle_motion = p('settle_motion_deg', 0.2).value     # 'no motion' threshold
        # Feedforward: the servos stop at a repeatable offset from the target (gravity sag
        # of J2-J4 + stopping short in the direction of motion). Every command of a
        # trajectory gets bias = ff_bias_deg + ff_dir_deg * sign(final - start) so the arm
        # usually ends within settle_tolerance without a correction. All zero = off.
        self.ff_bias = np.array(p('ff_bias_deg', [0.0] * 6).value, dtype=float)
        self.ff_dir = np.array(p('ff_dir_deg', [0.0] * 6).value, dtype=float)
        self.ff_min_move = p('ff_min_move_deg', 1.0).value         # smaller moves: no dir term
        if self.ff_bias.shape != (6,) or self.ff_dir.shape != (6,):
            raise ValueError('ff_bias_deg and ff_dir_deg must be 6 values')
        self.gripper_joint = p('gripper_joint', 'gripper_controller').value
        self.gripper_open_pos = p('gripper_open_position', 1.0).value    # joint value = 100
        self.gripper_closed_pos = p('gripper_closed_position', 0.0).value  # joint value = 0
        self.gripper_speed = p('gripper_speed', 50).value
        self.gripper_tol = p('gripper_tolerance', 5).value       # 0..100 units
        self.gripper_timeout = p('gripper_timeout', 5.0).value
        self.reconnect_after = p('reconnect_after_failures', 20).value
        # Safety: no successful get_angles for this long -> stop the arm, abort the trajectory,
        # stop publishing joint_states (a frozen state lets MoveIt plan from a wrong start).
        self.state_timeout = p('state_timeout', 1.0).value      # [s]
        fake_vel = p('fake_max_vel_deg', 120.0).value
        fake_short = p('fake_stop_short_deg', [0.0] * 6).value   # fake servos stop this short
        fake_sag = p('fake_sag_deg', [0.0] * 6).value            # fake servos end this far below

        if self.connection == 'fake':
            self.robot = FakeRobot(max_vel_deg=fake_vel, stop_short_deg=fake_short,
                                   sag_deg=fake_sag)
        elif self.connection in ('socket', 'serial'):
            self.robot = PyMyCobotRobot(self.connection, ip=ip, netport=netport,
                                        serial_port=serial_port, baudrate=baudrate,
                                        gripper_id=gripper_id, use_gripper=self.use_gripper,
                                        logger=self.get_logger())
        else:
            raise ValueError(f'connection must be fake|socket|serial, got {self.connection!r}')

        where = {'socket': f'{ip}:{netport}', 'serial': f'{serial_port}@{baudrate}',
                 'fake': 'simulation'}[self.connection]
        self.get_logger().info(f'connecting ({self.connection}: {where}) ...')
        self.robot.connect()

        self._lock = threading.Lock()
        self._q = None                 # latest arm angles [deg]
        self._g = None                 # latest gripper 0..100
        self._traj = None
        self._grip = None
        self._requests = []
        self._fail_count = 0
        self._q_t = None               # monotonic time of the last successful angle read
        self._read_fails = 0           # consecutive failed angle reads
        self._stale_logged = False
        self._grip_fails = 0
        self._wait_first_state()
        self._warn_if_outside_limits()

        cb = ReentrantCallbackGroup()
        self.js_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.arm_server = ActionServer(
            self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory',
            execute_callback=self._exec_traj, goal_callback=self._goal_traj,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        self.grip_server = ActionServer(
            self, GripperCommand, 'gripper_controller/gripper_cmd',
            execute_callback=self._exec_gripper,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cb)
        for name, fn in (('power_on', self.robot.power_on),
                         ('release_servos', self.robot.release_servos),
                         ('focus_servos', self.robot.focus_servos),
                         ('stop', self._stop_all)):
            self.create_service(Trigger, f'~/{name}', self._make_srv(name, fn), callback_group=cb)

        self._running = True
        self._thread = threading.Thread(target=self._hw_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f'ready: joints={np.round(self._q, 1).tolist()} gripper={self._g} '
            f'loop {self.loop_rate:.0f} Hz, max {self.max_joint_vel:.0f} deg/s')
        if np.any(self.signs != 1.0) or np.any(self.offsets != 0.0):
            self.get_logger().info(f'joint mapping: signs={self.signs.tolist()} '
                                   f'offsets_deg={self.offsets.tolist()}')

    # ------------------------------------------------------- unit conversion
    def _read_angles(self):
        """Robot angles -> URDF frame [deg] (np.array) or None."""
        q = self.robot.get_angles()
        if q is None:
            return None
        return self.signs * np.array(q) + self.offsets

    def _send_angles(self, urdf_deg, speed=None):
        robot_deg = (np.asarray(urdf_deg) - self.offsets) * self.signs
        self.robot.send_angles(robot_deg.tolist(), self.stream_speed if speed is None else speed)

    # ------------------------------------------------------------------ setup
    def _wait_first_state(self):
        for _ in range(20):
            t = time.monotonic()
            q = self._read_angles()
            dt = time.monotonic() - t
            if q is not None:
                self._q = q
                self._q_t = time.monotonic()
                self.get_logger().info(f'get_angles round trip {dt * 1000:.0f} ms')
                break
            time.sleep(0.2)
        else:
            raise RuntimeError('could not read joint angles from the robot')
        if self.use_gripper:
            self._g = self.robot.get_gripper()
            self._grip_fails = 0 if self._g is not None else 3
            if self._g is None:
                self.get_logger().warn('could not read gripper angle (use_gripper:=false if '
                                       'no force-control gripper is attached)')

    def _warn_if_outside_limits(self):
        for j, (q, lo, hi) in enumerate(zip(self._q, self.j_min, self.j_max)):
            if q < lo - self.limit_margin or q > hi + self.limit_margin:
                self.get_logger().warn(
                    f'{ARM_JOINTS[j]} is at {q:.1f} deg, outside [{lo:.0f}, {hi:.0f}] deg. '
                    f'MoveIt will refuse to plan from here; move it back by hand '
                    f'(~/release_servos) or with a direct trajectory towards the range.')

    def _make_srv(self, name, fn):
        def cb(_req, res):
            r = Request(fn, name)
            with self._lock:
                self._requests.append(r)
            if not r.done.wait(5.0):
                res.success, res.message = False, 'timeout'
            else:
                res.success, res.message = r.ok, r.msg
            return res
        return cb

    def _stop_all(self):
        with self._lock:
            if self._traj is not None:
                self._traj.cancel = True
            if self._grip is not None:
                self._grip.cancel = True
        self.robot.stop()

    # -------------------------------------------------------------- hw thread
    def _hw_loop(self):
        period = 1.0 / self.loop_rate
        tick = 0
        log_t = 0.0
        while self._running and rclpy.ok():
            t_loop = time.monotonic()
            try:
                self._hw_step(tick)
                self._fail_count = 0
            except Exception as e:  # noqa: BLE001  (socket/serial errors)
                self._fail_count += 1
                self._read_fails = max(self._read_fails, 1)   # no good read this tick
                self._check_stale(self._traj)
                if t_loop - log_t > 2.0:
                    self.get_logger().error(f'hardware error: {e!r}')
                    log_t = t_loop
                if self._fail_count >= self.reconnect_after and self.connection != 'fake':
                    self._reconnect()
            tick += 1
            rest = period - (time.monotonic() - t_loop)
            if rest > 0:
                time.sleep(rest)

    def _reconnect(self):
        self.get_logger().warn('reconnecting ...')
        with self._lock:
            for job in (self._traj, self._grip):
                if job is not None:
                    job.cancel = True
        try:
            self.robot.close()
            self.robot.connect()
            self._fail_count = 0
            self.get_logger().info('reconnected')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'reconnect failed: {e!r}')
            time.sleep(1.0)

    def _hw_step(self, tick):
        with self._lock:
            requests, self._requests = self._requests, []
            traj, grip = self._traj, self._grip

        for r in requests:
            try:
                r.fn()
                r.ok, r.msg = True, f'{r.name} done'
            except Exception as e:  # noqa: BLE001
                r.ok, r.msg = False, repr(e)
            r.done.set()

        now = time.monotonic()
        if traj is not None:
            self._step_traj(traj, now)
        if grip is not None:
            self._step_gripper(grip, now)

        executing = traj is not None and not traj.done.is_set()
        if not executing or tick % self.read_divider == 0:
            q = None
            try:
                q = self._read_angles()
            finally:                  # exceptions count as failed reads too
                if q is not None:
                    self._q, self._q_t = q, time.monotonic()
                    if self._read_fails or self._stale_logged:
                        self.get_logger().info(
                            f'get_angles recovered after {self._read_fails} failed reads')
                    self._read_fails, self._stale_logged = 0, False
                else:
                    self._read_fails += 1
                    self._check_stale(traj)
        # An absent / silent gripper times out and its late replies desync the connection:
        # after 3 failed reads poll it only while a gripper command runs.
        g_div = 2 if grip is not None else self.gripper_read_divider
        poll_gripper = grip is not None or self._grip_fails < 3
        if self.use_gripper and poll_gripper and tick % g_div == 0:
            g = self.robot.get_gripper()
            if g is not None:
                self._g, self._grip_fails = g, 0
            else:
                self._grip_fails += 1
                if self._grip_fails == 3:
                    self.get_logger().warn('gripper not answering: idle gripper polling '
                                           'stopped (use_gripper:=false if none is attached)')
        self._publish_state()

        with self._lock:
            if self._traj is not None and self._traj.done.is_set():
                self._traj = None
            if self._grip is not None and self._grip.done.is_set():
                self._grip = None

    def _step_traj(self, job, now):
        if job.done.is_set():
            return
        if job.cancel:
            job.finish(FJT.PATH_TOLERANCE_VIOLATED,
                       'preempted' if job.preempted else 'canceled')
            if not job.preempted:
                self.robot.stop()
            return
        if job.t_start is None:
            job.t_start = now
        t = now - job.t_start
        job.desired = job.sample(t)

        if t < job.duration:
            ref = job.sample(t + self.lookahead)
            target = self._clip_cmd(ref + job.bias, ref)
            if job.last_sent is None or np.max(np.abs(target - job.last_sent)) >= self.min_step_deg:
                self._send_angles(target)
                job.last_sent = target
            return

        final = job.positions[-1]
        if not job.final_sent:
            job.cmd = self._clip_cmd(final + job.bias, final)
            self._send_angles(job.cmd)
            job.final_sent = True
            job.final_sent_t = now
            job.last_sent = job.cmd
            return
        if self._q is None:
            return
        q = self._q
        err = final - q
        emax = float(np.max(np.abs(err)))
        if emax <= self.settle_tol:
            self._finish_goal(job, emax)
            return
        # settled = the joints have not moved for settle_time since the last command
        if job.ref_q is None or np.max(np.abs(q - job.ref_q)) > self.settle_motion:
            job.ref_q, job.ref_t = q.copy(), now
        settled = (now - job.ref_t >= self.settle_time and
                   now - job.final_sent_t >= self.settle_time)
        if t > job.duration + self.goal_timeout:
            self._finish_goal(job, emax, timeout=True)
        elif settled:
            if job.corrections < self.max_corrections:
                if job.err0 is None:
                    job.err0 = emax
                corr = np.where(np.abs(err) > self.corr_deadband, err, 0.0)
                job.cmd = self._clip_cmd(job.cmd + corr, final)
                self._send_angles(job.cmd, self.corr_speed)
                job.corrections += 1
                job.final_sent_t = now
                job.ref_q = None
            else:
                self._finish_goal(job, emax)

    def _clip_cmd(self, cmd, ref):
        """Keep a command within max_correction_deg of the trajectory point ref and in the
        limits (ref itself may lie outside them, see _build_job)."""
        cmd = np.clip(cmd, ref - self.max_corr, ref + self.max_corr)
        return np.clip(cmd, np.minimum(self.j_min, ref), np.maximum(self.j_max, ref))

    def _finish_goal(self, job, emax, timeout=False):
        how = (f'{emax:.1f} deg after {job.corrections} correction(s) (was {job.err0:.1f})'
               if job.corrections else f'{emax:.1f} deg')
        if emax <= self.goal_tol:
            if job.corrections or np.any(job.bias):
                self.get_logger().info(f'goal reached: max joint error {how}')
            job.finish(FJT.SUCCESSFUL)
        else:
            job.finish(FJT.GOAL_TOLERANCE_VIOLATED,
                       f'goal not reached{" (timeout)" if timeout else ""}: max joint error '
                       f'{how} > {self.goal_tol} deg')

    def _step_gripper(self, job, now):
        if job.done.is_set():
            return
        if job.cancel:
            job.finish(False, msg='canceled')
            return
        if not job.sent:
            self.robot.set_gripper(job.target, self.gripper_speed)
            job.sent = True
            job.t_start = now
            return
        elapsed = now - job.t_start
        if self._g is not None and abs(self._g - job.target) <= self.gripper_tol:
            job.finish(True, reached=True)
            return
        if elapsed > 0.5:
            status = self.robot.get_gripper_status()
            if status == 2:
                job.finish(True, stalled=True, msg='object grasped')
                return
            if status == 1 and elapsed > 1.0:
                job.finish(True, stalled=True, msg='gripper stopped before target')
                return
        if elapsed > self.gripper_timeout:
            job.finish(False, msg='gripper timeout')

    def state_age(self):
        return float('inf') if self._q_t is None else time.monotonic() - self._q_t

    def is_stale(self):
        """Reads are failing and the last good one is older than state_timeout (a slow but
        working link is not stale: only failed reads count)."""
        return self._read_fails > 0 and self.state_age() > self.state_timeout

    def _check_stale(self, traj):
        """Joint angles unreadable for state_timeout: the arm may be moving while the
        published state is frozen -> stop it and abort the trajectory."""
        if not self.is_stale():
            return
        age = self.state_age()
        if traj is not None and not traj.done.is_set():
            traj.finish(FJT.PATH_TOLERANCE_VIOLATED,
                        f'joint angles not readable for {age:.1f} s '
                        f'({self._read_fails} failed reads): arm stopped')
            try:
                self.robot.stop()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'stop after stale state failed: {e!r}')
        if not self._stale_logged:
            self._stale_logged = True
            self.get_logger().error(
                f'get_angles failing ({self._read_fails} reads, {age:.1f} s): joint_states '
                'not published, trajectories rejected until it recovers')

    def _publish_state(self):
        if self._q is None or self.is_stale():
            return                    # never publish a frozen state
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(ARM_JOINTS)
        js.position = [math.radians(a) for a in self._q]
        if self.use_gripper:
            g = self._g if self._g is not None else 100
            js.name.append(self.gripper_joint)
            js.position.append(self._gripper_to_joint(g))
        self.js_pub.publish(js)

    def _gripper_to_joint(self, g):
        return self.gripper_closed_pos + (self.gripper_open_pos - self.gripper_closed_pos) * g / 100.0

    def _joint_to_gripper(self, pos):
        span = self.gripper_open_pos - self.gripper_closed_pos
        return float(np.clip((pos - self.gripper_closed_pos) / span * 100.0, 0.0, 100.0))

    # ------------------------------------------------------------ arm action
    def _goal_traj(self, goal):
        unknown = [n for n in goal.trajectory.joint_names if n not in ARM_JOINTS]
        if unknown or not goal.trajectory.points:
            self.get_logger().error(f'rejecting trajectory: unknown joints {unknown} '
                                    f'or no points')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _build_job(self, traj):
        if self.is_stale():
            return None, (f'joint state is stale ({self.state_age():.1f} s, get_angles failing): '
                          'refusing to move')
        q_now = self._q.copy()
        idx = [traj.joint_names.index(n) if n in traj.joint_names else None for n in ARM_JOINTS]
        times, pos = [], []
        for pt in traj.points:
            row = [math.degrees(pt.positions[i]) if i is not None else q_now[j]
                   for j, i in enumerate(idx)]
            times.append(_dur(pt.time_from_start))
            pos.append(row)
        times, pos = np.array(times), np.array(pos)

        # If the arm is already outside a limit (e.g. moved by hand), allow motions that do
        # not go further out, so it can be brought back; otherwise use the limits as is.
        j_min = np.minimum(self.j_min, q_now) - self.limit_margin
        j_max = np.maximum(self.j_max, q_now) + self.limit_margin
        bad = (pos < j_min) | (pos > j_max)
        if np.any(bad):
            j = int(np.argmax(np.any(bad, axis=0)))
            return None, (f'{ARM_JOINTS[j]} goes to {pos[:, j].min():.1f}..{pos[:, j].max():.1f} deg, '
                          f'outside the limit [{self.j_min[j]:.0f}, {self.j_max[j]:.0f}] deg')
        pos = np.clip(pos, np.minimum(self.j_min, q_now), np.maximum(self.j_max, q_now))

        start_err = np.max(np.abs(pos[0] - q_now))
        if times[0] <= 1e-6 and start_err > self.start_tol:
            return None, (f'trajectory start is {start_err:.1f} deg from the current state '
                          f'(start_tolerance_deg {self.start_tol})')
        if times[0] > 1e-6:          # trajectory does not start at t=0: start from here
            times = np.insert(times, 0, 0.0)
            pos = np.vstack([q_now, pos])
        if np.any(np.diff(times) < 0):
            return None, 'time_from_start is not increasing'

        dt = np.diff(times)
        dq = np.abs(np.diff(pos, axis=0))
        vmax = float(np.max(dq / np.maximum(dt, 1e-3)[:, None])) if len(dt) else 0.0
        scale = max(1.0, vmax / self.max_joint_vel) if self.max_joint_vel > 0 else 1.0
        if scale > 1.0 + 1e-3:
            self.get_logger().warn(
                f'trajectory peak {vmax:.0f} deg/s > max_joint_vel_deg {self.max_joint_vel:.0f}: '
                f'slowing down x{scale:.2f} (lower MoveIt velocity scaling to avoid this)')
        if len(times) == 1:          # single point: move there at max_joint_vel
            d = float(np.max(np.abs(pos[0] - q_now)))
            times = np.array([0.0, max(d / self.max_joint_vel, 0.1)])
            pos = np.vstack([q_now, pos])
        job = TrajJob(times * scale, pos)
        move = pos[-1] - pos[0]
        job.bias = self.ff_bias + self.ff_dir * np.where(np.abs(move) >= self.ff_min_move,
                                                         np.sign(move), 0.0)
        return job, ''

    def _exec_traj(self, goal_handle):
        result = FollowJointTrajectory.Result()
        traj = goal_handle.request.trajectory
        job, err = self._build_job(traj)
        if job is None:
            self.get_logger().error(err)
            goal_handle.abort()
            result.error_code, result.error_string = FJT.INVALID_GOAL, err
            return result

        with self._lock:
            if self._traj is not None:
                self._traj.preempted = True
                self._traj.cancel = True
                self._traj.finish(FJT.PATH_TOLERANCE_VIOLATED, 'preempted')
            self._traj = job
        self.get_logger().info(f'executing trajectory: {len(job.times)} points, '
                               f'{job.duration:.2f} s')

        fb = FollowJointTrajectory.Feedback()
        fb.joint_names = list(ARM_JOINTS)
        while not job.done.wait(0.1):
            if goal_handle.is_cancel_requested:
                job.cancel = True
                continue
            fb.header.stamp = self.get_clock().now().to_msg()
            des = np.radians(job.desired)
            act = np.radians(self._q)
            fb.desired = JointTrajectoryPoint(positions=des.tolist())
            fb.actual = JointTrajectoryPoint(positions=act.tolist())
            fb.error = JointTrajectoryPoint(positions=(des - act).tolist())
            goal_handle.publish_feedback(fb)

        result.error_code, result.error_string = job.error_code, job.error_string
        if goal_handle.is_cancel_requested and not job.preempted:
            goal_handle.canceled()
            self.get_logger().info('trajectory canceled')
        elif job.error_code == FJT.SUCCESSFUL:
            goal_handle.succeed()
            self.get_logger().info('trajectory done')
        else:
            goal_handle.abort()
            self.get_logger().warn(f'trajectory aborted: {job.error_string}')
        return result

    # -------------------------------------------------------- gripper action
    def _exec_gripper(self, goal_handle):
        result = GripperCommand.Result()
        if not self.use_gripper:
            goal_handle.abort()
            return result
        target = self._joint_to_gripper(goal_handle.request.command.position)
        job = GripperJob(target)
        with self._lock:
            if self._grip is not None:
                self._grip.finish(False, msg='preempted')
            self._grip = job
        while not job.done.wait(0.1):
            if goal_handle.is_cancel_requested:
                job.cancel = True
        g = self._g if self._g is not None else target
        result.position = self._gripper_to_joint(g)
        result.reached_goal, result.stalled = job.reached, job.stalled
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif job.ok:
            goal_handle.succeed()
            self.get_logger().info(f'gripper -> {target:.0f} ({job.msg or "reached"}), now {g}')
        else:
            goal_handle.abort()
            self.get_logger().warn(f'gripper: {job.msg}')
        return result

    def destroy_node(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2.0)
        moving = self._traj is not None and not self._traj.done.is_set()
        if moving and self.connection != 'fake':
            try:                      # do not leave the arm running towards the last setpoint
                self.robot.stop()
            except Exception:  # noqa: BLE001
                pass
        self.robot.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MyCobotDriver()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
