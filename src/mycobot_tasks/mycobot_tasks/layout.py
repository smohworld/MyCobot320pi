"""Layout simulation in RViz: drag the desk (where the robot sits on it), the TurtleBot3 Burger
(its box) and the SISYPHUS shelf robot, move the shelf axes, and see the planning scene
follow. Saved to ~/.ros/layout.yaml (place_scene --layout reads it).

  ros2 launch mycobot_calibration handeye.launch.py sim:=true scene_args:="--layout"
  ros2 run mycobot_tasks layout          # RViz: InteractiveMarkers display, namespace /layout,
                                         # Interact tool (i); MotionPlanning scene shows boxes
  ros2 run mycobot_tasks layout --print  # numbers + the exported files, no ROS
  ros2 run mycobot_tasks layout --list                 # saved layouts (~/final_project_ws/layouts/*.yaml)
  ros2 run mycobot_tasks layout --save-as NAME         # the current layout -> layouts/NAME.yaml
  ros2 run mycobot_tasks layout --load NAME [--print]  # start from layouts/NAME.yaml
  RViz right click > layouts: save as <date-time> / load: NAME (while running)

Handles (base frame; the robot base stays at the origin, the rest moves around it):
  robot     red arrow ring over the base: turn = the robot's direction on the desk; RViz's
            fixed frame is the robot base, so the desk, TurtleBot and shelf turn around it
            together (the support plates stay on the robot) (--robot-yaw)
  desk      disc at the desk top centre: drag = move the desk under the robot, ring = turn
            the desk alone (place_scene --desk-x / --edge-y / --robot-yaw)
  TurtleBot box centre: drag / turn (place_scene --bot cx,cy,yaw; marker = box floor centre)
  shelf     base plate centre on the floor: drag / turn (place_scene --shelf, corner kept)
  fork      on the D11 fork top: red = X axis (--shelf-x, +-97 mm), green = fork out (--fork-y,
            0..160), blue = fork height (--fork-top, 263..616 mm over the floor)
  tray      yellow box over D11: the tray on the fork (drag / turn; +-5 cm, +-20 deg off the
            fork centre): the exported tray file follows (layout_tray.yaml)
  lock      right click > lock / unlock desk + shelf: the desk, shelf and fork handles (and
            the robot ring) stop moving; the robot handle then moves the robot across the
            desk (y) instead; the tray on D11 and the TurtleBot stay movable
  right click: D11 height (desk top / lowest / highest), tray direction on D11, show the
            tray, arm (simulation): go above D11 / the TurtleBot box / back to the start
            pose (fake driver only; --above sets the height), snap yaw, reset.

The search regions (~/.ros/regions.yaml, the `regions` boxes) follow: shelf region on D11,
bot region on the TurtleBot box, sizes kept (--no-sync-regions = off; the file before the
first change is kept as regions.yaml.bak-before-layout).

Every change also writes the files for a planned run with the real planner (simulation):
  ~/.ros/layout_tray.yaml      tray on D11 (handle top = fork top + --tray-height)
  ~/.ros/layout_tray_bot.yaml  tray in the TurtleBot box
  ~/.ros/layout_bot.yaml       TurtleBot marker (box floor centre)
  ~/.ros/layout_regions.yaml   shelf region = D11 (z = fork top), bot region = the box
D11 -> TurtleBot (m):
  ros2 run mycobot_tasks tray_handle_pick --replay ~/.ros/layout_tray.yaml \\
      --bot-save ~/.ros/layout_bot.yaml --regions ~/.ros/layout_regions.yaml \\
      --scene-args=--layout --go --place-at bot --place-z auto --place-yaw bot
TurtleBot -> D11 (k): the same with --replay ~/.ros/layout_tray_bot.yaml
Restart tray_handle_pick after moving things (it reads the files at start).
"""
import argparse
import math
import os
import sys
import time

import yaml

from mycobot_tasks import place_scene as ps

PRESETS = os.environ.get('LAYOUTS_DIR', '~/final_project_ws/layouts')   # saved layouts <name>.yaml
FILES = dict(tray='~/.ros/layout_tray.yaml', tray_bot='~/.ros/layout_tray_bot.yaml',
             bot='~/.ros/layout_bot.yaml', regions='~/.ros/layout_regions.yaml')
D11_Y = (287.0 + 427.0) / 2 / 1000           # D11 centre, shelf Y with the fork in [m]
PLATE_C = (0.300, 0.225)                     # base plate centre, shelf frame [m]
FORK_Y = (0.0, 0.16)
# default placement (2026-11-20): shelf yaw +90 beside the +y desk edge, its left side
# (X = 0) toward the robot, no box overlap with the desk / legs; D11 (+0.003, +0.313),
# r 31.3 cm, lowest fork (263 mm): top-down IK reaches ~34 cm at 2 cm over the handle,
# ~30 cm at 8 cm -> approach from --above 0.04
# desk / TurtleBot as set in the layout 2026-11-20 (robot 9.4 cm from the +y desk edge; the
# shelf above is only clear of THIS desk; place_scene defaults = these since 2026-11-26)
DEFAULTS = dict(desk_x=[-1.0344, 0.0656], edge_y=-0.5062, robot_yaw=-0.0418,
                shelf=[0.36, 0.11, 90.0], with_shelf=True, shelf_x=-0.097, fork_top=0.263,
                fork_y=0.0, with_bot=True, bot=[0.2155, -0.0449, 180.0], tray_yaw_off=90.0,
                show_tray=False, locked=False, tray_dx=0.0, tray_dy=0.0, tray_dyaw=0.0)
# not place_scene options, kept in the layout file too
EXTRA_KEYS = ('tray_yaw_off', 'show_tray', 'locked', 'tray_dx', 'tray_dy', 'tray_dyaw')
TRAY_OFF_MAX, TRAY_DYAW_MAX = 0.05, 20.0      # tray on D11: off the fork centre [m], [deg]


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


class Layout:
    """The layout values + the place_scene args built from them."""

    def __init__(self, path):
        self.path = os.path.expanduser(path)
        self.base = ps.parse([])                       # place_scene defaults
        self.v = self.defaults()
        try:
            with open(self.path) as f:
                d = yaml.safe_load(f) or {}
            self.v.update({k: d[k] for k in self.v if k in d and d[k] is not None})
        except (OSError, yaml.YAMLError):
            pass
        if self.v['bot'] is None:
            self.v['bot'] = self.default_bot()

    def load_file(self, path):
        """Values from a layout file (e.g. a saved preset) over the defaults."""
        with open(os.path.expanduser(path)) as f:
            d = yaml.safe_load(f) or {}
        self.v = self.defaults()
        self.v.update({k: d[k] for k in self.v if k in d and d[k] is not None})
        if self.v['bot'] is None:
            self.v['bot'] = self.default_bot()

    def save_as(self, name, folder=PRESETS):
        """Save the current layout as a preset <folder>/<name>.yaml -> its path."""
        folder = os.path.expanduser(folder)
        os.makedirs(folder, exist_ok=True)
        keep, self.path = self.path, os.path.join(folder, f'{name}.yaml')
        try:
            self.save()
            return self.path
        finally:
            self.path = keep

    def defaults(self):
        b = self.base
        d = dict(desk_x=list(b.desk_x), edge_y=b.edge_y, robot_yaw=b.robot_yaw)
        d.update({k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()})
        return d

    def default_bot(self):
        th = math.radians(self.v['robot_yaw'])
        x, y = ps.rot(-th, 0.0, self.v['edge_y'] - 0.15)
        return [x, y, math.degrees(math.atan2(y, x))]

    def args(self, with_tray=None):
        a = ps.parse(SCENE_EXTRA)
        for k in ps.LAYOUT_KEYS:
            setattr(a, k, self.v[k])
        a.with_tray = self.v['show_tray'] if with_tray is None else with_tray
        a.tray_dx = self.v['tray_dx']                    # the tray off the D11 centre
        a.tray_y = ps.parse([]).tray_y + self.v['tray_dy']
        return a

    def save(self):
        out = {k: self.v[k] for k in ps.LAYOUT_KEYS}
        out.update({k: self.v[k] for k in EXTRA_KEYS})
        out = {k: ([round(float(t), 4) for t in v] if isinstance(v, list) else
                   (round(v, 4) if isinstance(v, float) else v)) for k, v in out.items()}
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            f.write(f"# ros2 run mycobot_tasks layout, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            yaml.safe_dump(out, f, sort_keys=False)
        os.replace(tmp, self.path)

    # geometry (base frame)
    @property
    def floor(self):
        return self.base.table_z - self.base.desk_height

    def shelf_to_base(self, X, Y):
        ox, oy, yd = self.v['shelf']
        p = ps.rot(math.radians(yd), X, Y)
        return ox + p[0], oy + p[1]

    def base_to_shelf(self, x, y):
        ox, oy, yd = self.v['shelf']
        return tuple(ps.rot(-math.radians(yd), x - ox, y - oy))

    def fork_point(self):
        """D11 top centre (base xyz)."""
        x, y = self.shelf_to_base(PLATE_C[0] + self.v['shelf_x'], D11_Y + self.v['fork_y'])
        return x, y, self.floor + self.v['fork_top']

    def plate_centre(self):
        return self.shelf_to_base(*PLATE_C)

    def set_plate_centre(self, x, y, yaw_deg):
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        X, Y = PLATE_C
        self.v['shelf'] = [x - (c * X - s * Y), y - (s * X + c * Y), yaw_deg]

    def desk_pose(self):
        """Desk top centre (base xy) and its yaw in base [deg]."""
        th = math.radians(self.v['robot_yaw'])
        cx = sum(self.v['desk_x']) / 2
        cy = self.v['edge_y'] + self.base.desk_width / 2
        x, y = ps.rot(-th, cx, cy)
        return x, y, -self.v['robot_yaw']

    def set_desk_pose(self, x, y, yaw_deg):
        th = -math.radians(yaw_deg)
        cx, cy = ps.rot(th, x, y)
        half = (self.v['desk_x'][1] - self.v['desk_x'][0]) / 2
        self.v['desk_x'] = [cx - half, cx + half]
        self.v['edge_y'] = cy - self.base.desk_width / 2
        self.v['robot_yaw'] = math.degrees(th)

    def turn_world(self, d_deg):
        """The robot turned on the desk by -d_deg: desk, TurtleBot and shelf turn by d_deg
        about the base axis (the base frame is fixed; the support plates are on the robot)."""
        x, y, yd = self.desk_pose()
        px, py = ps.rot(math.radians(d_deg), x, y)
        self.set_desk_pose(px, py, yd + d_deg)
        bx, by, byaw = self.v['bot']
        px, py = ps.rot(math.radians(d_deg), bx, by)
        self.v['bot'] = [px, py, math.remainder(byaw + d_deg, 360.0)]
        x, y = self.plate_centre()
        px, py = ps.rot(math.radians(d_deg), x, y)
        self.set_plate_centre(px, py, math.remainder(self.v['shelf'][2] + d_deg, 360.0))

    def set_fork(self, x, y, z):
        X, Y = self.base_to_shelf(x, y)
        self.v['shelf_x'] = clamp(X - PLATE_C[0], -ps.X_TRAVEL / 1000, ps.X_TRAVEL / 1000)
        self.v['fork_y'] = clamp(Y - D11_Y, *FORK_Y)
        self.v['fork_top'] = clamp(z - self.floor, ps.FORK_TOP[0] / 1000, ps.FORK_TOP[1] / 1000)

    def desk_y_axis(self):
        """The desk's y axis (across the desk) in base."""
        return ps.rot(-math.radians(self.v['robot_yaw']), 0.0, 1.0)

    def shift_robot(self, d):
        """The robot moved d along the desk y axis (desk, shelf and TurtleBot stay in the
        room): in the base frame they all move by -d."""
        ux, uy = self.desk_y_axis()
        self.v['edge_y'] -= d
        x, y = self.plate_centre()
        self.set_plate_centre(x - d * ux, y - d * uy, self.v['shelf'][2])
        bx, by, byaw = self.v['bot']
        self.v['bot'] = [bx - d * ux, by - d * uy, byaw]

    def tray_pose(self):
        """The tray handle on D11 (base xyz of the handle midpoint at the fork top) and its
        L -> R yaw [rad]: the fork centre + (tray_dx, tray_dy) in the shelf frame."""
        fx, fy, fz = self.fork_point()
        dx, dy = ps.rot(math.radians(self.v['shelf'][2]), self.v['tray_dx'], self.v['tray_dy'])
        yaw = math.radians(self.v['shelf'][2] + self.v['tray_yaw_off'] + self.v['tray_dyaw'])
        return fx + dx, fy + dy, fz, math.remainder(yaw, 2 * math.pi)

    def set_tray(self, x, y, yaw_deg):
        fx, fy, _ = self.fork_point()
        X, Y = self.base_to_shelf(x, y)
        Xf, Yf = self.base_to_shelf(fx, fy)
        m = TRAY_OFF_MAX
        self.v['tray_dx'], self.v['tray_dy'] = clamp(X - Xf, -m, m), clamp(Y - Yf, -m, m)
        d = math.remainder(yaw_deg - self.v['shelf'][2] - self.v['tray_yaw_off'], 360.0)
        self.v['tray_dyaw'] = clamp(d, -TRAY_DYAW_MAX, TRAY_DYAW_MAX)

    def box_floor_z(self):
        return self.floor + self.base.box_floor

    def summary(self):
        fx, fy, fz = self.fork_point()
        bx, by, byaw = self.v['bot']
        dx, dy, dyaw = self.desk_pose()
        return (f"desk centre ({dx:+.3f}, {dy:+.3f}) yaw {dyaw:+.1f} deg "
                f"[--desk-x {self.v['desk_x'][0]:.3f},{self.v['desk_x'][1]:.3f} --edge-y "
                f"{self.v['edge_y']:.3f} --robot-yaw {self.v['robot_yaw']:.1f}]\n"
                f"TurtleBot box ({bx:+.3f}, {by:+.3f}) yaw {byaw:+.1f} deg, "
                f"{math.hypot(bx, by) * 100:.0f} cm from the base axis, floor z "
                f"{self.box_floor_z():+.3f}\n"
                f"shelf corner ({self.v['shelf'][0]:+.3f}, {self.v['shelf'][1]:+.3f}) yaw "
                f"{self.v['shelf'][2]:+.1f} deg; X {self.v['shelf_x'] * 1000:+.0f} mm, fork top "
                f"{self.v['fork_top'] * 1000:.0f} mm over the floor, fork out "
                f"{self.v['fork_y'] * 1000:.0f} mm\n"
                f"D11 top ({fx:+.3f}, {fy:+.3f}, {fz:+.3f}), {math.hypot(fx, fy) * 100:.0f} cm "
                f"from the base axis; tray on it {self.v['tray_dx'] * 1000:+.0f} / "
                f"{self.v['tray_dy'] * 1000:+.0f} mm (shelf X / Y), yaw "
                f"{self.v['tray_dyaw']:+.0f} deg" + ('; desk + shelf LOCKED' if self.v['locked'] else ''))

    def sync_regions(self, path):
        """The search regions (regions.yaml, `regions` node in RViz) follow the layout: the
        shelf region centred on D11 (z = D11 top), the bot region on the TurtleBot box (z = its
        floor); their sizes and camera heights are kept. The file before the first sync is
        kept as <path>.bak-before-layout. -> True when the file changed."""
        import shutil

        from mycobot_tasks import regions as rg
        path = os.path.expanduser(path)
        regions = rg.load(path)
        old = yaml.safe_dump(regions)
        fx, fy, fz = self.fork_point()
        bx, by, _ = self.v['bot']
        rg.move_to(regions['shelf'], fx, fy)
        regions['shelf']['z'] = fz
        rg.move_to(regions['bot'], bx, by)
        regions['bot']['z'] = self.box_floor_z()
        if yaml.safe_dump(regions) == old:
            return False
        bak = path + '.bak-before-layout'
        if os.path.exists(path) and not os.path.exists(bak):
            shutil.copy2(path, bak)
        rg.save(regions, path)
        return True

    def export(self, tray_height=0.078):
        """Replay / region files for tray_handle_pick (see the module doc)."""
        fx, fy, fz = self.fork_point()
        tx, ty, _, tray_yaw = self.tray_pose()
        bx, by, byaw = self.v['bot']
        bz = self.box_floor_z()
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')

        def det(x, y, surface, yaw):
            return dict(stamp=stamp, tray=0, handle_xyz=[x, y, surface + tray_height],
                        yaw_lr_deg=math.degrees(yaw), spacing=0.057, tilt_deg=0.0,
                        table_z=surface, camera_q_deg=None)
        files = {
            'tray': det(tx, ty, fz, tray_yaw),
            'tray_bot': det(bx, by, bz, math.radians(byaw)),
            'bot': dict(stamp=stamp, marker_id=0, marker_xyz=[bx, by, bz], yaw_deg=byaw,
                        tilt_deg=0.0),
            'regions': dict(
                shelf=dict(x=[fx - 0.05, fx + 0.05], y=[fy - 0.05, fy + 0.05], z=fz, view=0.28),
                bot=dict(x=[bx - 0.12, bx + 0.12], y=[by - 0.12, by + 0.12], z=bz, view=0.28)),
        }

        def rnd(v):
            if isinstance(v, dict):
                return {k: rnd(t) for k, t in v.items()}
            if isinstance(v, list):
                return [rnd(t) for t in v]
            return round(v, 4) if isinstance(v, float) else v
        for k, d in files.items():
            path = os.path.expanduser(FILES[k])
            with open(path + '.tmp', 'w') as f:
                yaml.safe_dump(rnd(d), f, sort_keys=False)
            os.replace(path + '.tmp', path)


def presets(folder=PRESETS):
    """Saved layout names (newest first)."""
    folder = os.path.expanduser(folder)
    try:
        fs = [f for f in os.listdir(folder) if f.endswith('.yaml')]
    except OSError:
        return []
    fs.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return [f[:-5] for f in fs]


def preset_path(name, folder=PRESETS):
    p = os.path.expanduser(name)
    if os.sep in name or name.endswith('.yaml'):
        return p
    return os.path.join(os.path.expanduser(folder), f'{name}.yaml')


# extra place_scene options not kept in the layout file (e.g. --rack-trays), from
# --scene-args: the layout node re-sends the scene on every change and removes the objects
# it does not build, so it needs the same options as place_scene
SCENE_EXTRA = []


def scene_extra(text):
    import shlex
    t, out, i = shlex.split(text), [], 0
    while i < len(t):
        if t[i] == '--layout':                  # the layout file is this node's own
            i += 2 if i + 1 < len(t) and not t[i + 1].startswith('-') else 1
            continue
        out.append(t[i])
        i += 1
    return out


def yaw_of(q):
    return math.degrees(2 * math.atan2(q.z, q.w))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--file', default=ps.LAYOUT)
    p.add_argument('--print', action='store_true', help='numbers + files only, no ROS')
    p.add_argument('--reset', action='store_true', help='start from the defaults')
    p.add_argument('--presets', default=PRESETS, help='folder of the saved layouts')
    p.add_argument('--load', metavar='NAME',
                   help='start from the saved layout NAME (--presets folder, or a path): it '
                        'becomes --file (and the exports); RViz menu: layouts > load')
    p.add_argument('--save-as', metavar='NAME',
                   help='save the layout (--file) as NAME in --presets and exit (no ROS); '
                        'RViz menu: layouts > save as ...')
    p.add_argument('--list', action='store_true', help='list the saved layouts and exit')
    p.add_argument('--tray-height', type=float, default=0.078,
                   help='tray bottom -> handle top for the exported tray files [m]')
    p.add_argument('--frame', default='base')
    p.add_argument('--above', type=float, default=0.04,
                   help="menu 'arm: go above ...': finger tips this far over the tray's handle "
                        'top (lowered in 2 cm steps to 2 cm when not reachable) [m]')
    p.add_argument('--regions', default='~/.ros/regions.yaml',
                   help='search regions file that follows the layout (shelf region on D11, '
                        'bot region on the TurtleBot box)')
    p.add_argument('--no-sync-regions', action='store_true',
                   help='leave the regions file alone')
    p.add_argument('--no-flip', action='store_true',
                   help="menu 'arm: go above ...': never use the reversed arm (J1 ~180 deg "
                        'off the target)')
    p.add_argument('--real', action='store_true',
                   help="allow the 'arm' menu to move a real (non-fake) robot")
    p.add_argument('--scene-args', default='',
                   help='place_scene options beyond the layout file, as one string, e.g. '
                        '"--rack-trays" (handeye.launch.py passes its scene_args)')
    args = p.parse_args(sys.argv[1:] if '--ros-args' not in sys.argv
                        else sys.argv[1:sys.argv.index('--ros-args')])
    global SCENE_EXTRA
    SCENE_EXTRA = scene_extra(args.scene_args)
    if args.list:
        for n in presets(args.presets):
            print(f'{n:30s} {time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(preset_path(n, args.presets))))}')
        return
    L = Layout(args.file)
    if args.load:
        L.load_file(preset_path(args.load, args.presets))
        print(f'loaded {preset_path(args.load, args.presets)}')
    if args.save_as:
        print(f'saved {L.save_as(args.save_as, args.presets)}')
        return
    if args.reset:
        L.v = L.defaults()
        if L.v['bot'] is None:
            L.v['bot'] = L.default_bot()
    if args.print:
        objs, info = ps.build(L.args())
        ps.report(objs, info, L.args())
        print(L.summary())
        L.save()
        L.export(args.tray_height)
        if not args.no_sync_regions and L.sync_regions(args.regions):
            print(f'regions moved: {args.regions}')
        print(f'saved {L.path} and ' + ', '.join(FILES.values()))
        return

    import rclpy
    from geometry_msgs.msg import Point
    from interactive_markers import InteractiveMarkerServer, MenuHandler
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from moveit_msgs.srv import ApplyPlanningScene
    from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                        InteractiveMarkerFeedback, Marker)

    rclpy.init()
    node = rclpy.create_node('layout')
    log = node.get_logger()
    server = InteractiveMarkerServer(node, 'layout')
    cli = node.create_client(ApplyPlanningScene, 'apply_planning_scene')
    state = dict(last=0.0, dirty=False, warned=False)

    def send(cos):
        if not cli.service_is_ready():
            if not state['warned']:
                log.warn('apply_planning_scene not available (move_group running?)')
                state['warned'] = True
            return
        state['warned'] = False
        req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
        req.scene.world.collision_objects = cos
        req.scene.object_colors = ps.object_colors(
            [co.id for co in cos if co.operation == CollisionObject.ADD])
        cli.call_async(req)

    def remove(names):
        for n in names:                    # one by one: removing a missing object fails
            co = CollisionObject()
            co.header.frame_id, co.id, co.operation = args.frame, n, CollisionObject.REMOVE
            send([co])

    def update_scene(force=False):
        now = time.monotonic()
        if not force and now - state['last'] < 0.15:
            state['dirty'] = True
            return
        state['last'], state['dirty'] = now, False
        objs, _ = ps.build(L.args())
        missing = [n for n in ps.NAMES if n not in objs]
        if missing:
            remove(missing)
        send(ps.collision_objects(objs, args.frame))   # ADD of an existing id replaces it

    def flush():
        if state['dirty']:
            update_scene(force=True)

    def store():
        L.save()
        L.export(args.tray_height)
        if not args.no_sync_regions and L.sync_regions(args.regions):
            log.info(f'regions moved with the layout ({args.regions}): shelf on D11, bot on '
                     'the TurtleBot box')
        log.info('saved\n' + L.summary())

    # ---- markers
    def control(name, mode, q, markers=()):
        c = InteractiveMarkerControl(always_visible=True, name=name, interaction_mode=mode)
        c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z = (float(t) for t in q)
        c.markers.extend(markers)
        return c

    UP = (0.7071068, 0.0, 0.7071068, 0.0)          # control axis = z (plane normal = z)

    def flat(rgb, sx, sy, a=0.5, h=0.006, cyl=False):
        m = Marker(type=Marker.CYLINDER if cyl else Marker.CUBE)
        m.scale.x, m.scale.y, m.scale.z = sx, sy, h
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = a
        return m

    def arrow(rgb, length=0.06):
        m = Marker(type=Marker.ARROW)
        m.scale.x, m.scale.y, m.scale.z = 0.008, 0.016, 0.02
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 0.9
        m.points = [Point(), Point(x=length)]
        return m

    def planar(name, x, y, z, yaw_deg, desc, vis, scale, fixed=False):
        im = InteractiveMarker()
        im.header.frame_id = args.frame
        im.name, im.description, im.scale = name, desc + ('  [locked]' if fixed else ''), scale
        im.pose.position.x, im.pose.position.y, im.pose.position.z = x, y, z
        im.pose.orientation.z = math.sin(math.radians(yaw_deg) / 2)
        im.pose.orientation.w = math.cos(math.radians(yaw_deg) / 2)
        if fixed:                                # shown, not movable (menu still works)
            im.controls.append(control('show', InteractiveMarkerControl.BUTTON, UP, vis))
            return im
        im.controls.append(control('move', InteractiveMarkerControl.MOVE_PLANE, UP, vis))
        im.controls.append(control('turn', InteractiveMarkerControl.ROTATE_AXIS, UP))
        return im

    def fork_marker():
        x, y, z = L.fork_point()
        im = InteractiveMarker()
        im.header.frame_id = args.frame
        im.name = 'fork'
        im.description = (f"D11  X {L.v['shelf_x'] * 1000:+.0f}  out {L.v['fork_y'] * 1000:.0f}"
                          f"  top {L.v['fork_top'] * 1000:.0f} mm ({z:+.3f})  "
                          f"r {math.hypot(x, y) * 100:.0f} cm")
        im.scale = 0.12
        im.pose.position.x, im.pose.position.y, im.pose.position.z = x, y, z + 0.002
        yd = L.v['shelf'][2]
        im.pose.orientation.z = math.sin(math.radians(yd) / 2)
        im.pose.orientation.w = math.cos(math.radians(yd) / 2)
        pad = flat((1.0, 0.55, 0.1), 0.10, 0.14, a=0.6, h=0.004)
        if L.v['locked']:
            im.description += '  [locked]'
            im.controls.append(control('pad', InteractiveMarkerControl.BUTTON, (1, 0, 0, 0), [pad]))
            return im
        im.controls.append(control('pad', InteractiveMarkerControl.NONE, (1, 0, 0, 0), [pad]))
        im.controls.append(control('shelf_x', InteractiveMarkerControl.MOVE_AXIS, (1, 0, 0, 0)))
        im.controls.append(control('fork_y', InteractiveMarkerControl.MOVE_AXIS,
                                   (0.7071068, 0.0, 0.0, 0.7071068)))
        im.controls.append(control('fork_top', InteractiveMarkerControl.MOVE_AXIS, UP))
        return im

    def robot_marker():
        yd = -L.v['robot_yaw']                   # = the desk direction in base
        im = InteractiveMarker()
        im.header.frame_id = args.frame
        im.name = 'robot'
        im.description = (f"robot direction: {L.v['robot_yaw']:+.1f} deg vs the desk "
                          "(turn the ring)")
        im.scale = 0.25
        im.pose.position.z = L.base.table_z + L.base.mount_up + 0.01
        im.pose.orientation.z = math.sin(math.radians(yd) / 2)
        im.pose.orientation.w = math.cos(math.radians(yd) / 2)
        if L.v['locked']:     # desk + shelf fixed: only the robot's place across the desk
            im.description = (f"robot: drag along the desk y (desk + shelf locked; robot "
                              f"{-L.v['edge_y'] * 1000:.0f} mm from the desk -y edge)")
            im.controls.append(control('shift_y', InteractiveMarkerControl.MOVE_AXIS,
                                       (0.7071068, 0.0, 0.0, 0.7071068),
                                       [arrow((0.9, 0.2, 0.2), 0.16)]))
            return im
        im.controls.append(control('turn', InteractiveMarkerControl.ROTATE_AXIS, UP,
                                   [arrow((0.9, 0.2, 0.2), 0.16)]))
        return im

    def tray_marker():
        x, y, z, yaw = L.tray_pose()
        r = math.hypot(x, y)
        return planar('tray', x, y, z + args.tray_height + 0.01, math.degrees(yaw),
                      f"tray on D11  X {L.v['tray_dx'] * 1000:+.0f}  Y "
                      f"{L.v['tray_dy'] * 1000:+.0f} mm  yaw {L.v['tray_dyaw']:+.0f}  r "
                      f"{r * 100:.1f} cm", [flat((1.0, 0.9, 0.15), 0.10, 0.15, a=0.5),
                                            arrow((1.0, 0.9, 0.15), 0.08)], 0.2)

    def publish(which=('robot', 'desk', 'bot', 'shelf', 'fork', 'tray')):
        if 'robot' in which:
            server.insert(robot_marker(), feedback_callback=feedback)
        if 'desk' in which:
            x, y, yd = L.desk_pose()
            server.insert(planar('desk', x, y, L.base.table_z + 0.015, yd,
                                 f"desk (robot at its front end)  yaw {yd:+.1f}",
                                 [flat((0.6, 0.45, 0.25), 0.10, 0.10, cyl=True),
                                  arrow((0.6, 0.45, 0.25))], 0.3, fixed=L.v['locked']),
                          feedback_callback=feedback)
        if 'bot' in which:
            bx, by, byaw = L.v['bot']
            server.insert(planar('bot', bx, by, L.box_floor_z() + 0.25, byaw,
                                 f"TurtleBot  r {math.hypot(bx, by) * 100:.0f} cm  yaw {byaw:+.0f}",
                                 [flat((0.2, 0.4, 1.0), 0.20, 0.16), arrow((0.2, 0.4, 1.0), 0.12)],
                                 0.3), feedback_callback=feedback)
        if 'shelf' in which:
            x, y = L.plate_centre()
            server.insert(planar('shelf', x, y, L.floor + 0.02, L.v['shelf'][2],
                                 f"shelf  yaw {L.v['shelf'][2]:+.0f}",
                                 [flat((0.1, 0.8, 0.2), 0.12, 0.12, cyl=True),
                                  arrow((0.1, 0.8, 0.2), 0.12)], 0.45, fixed=L.v['locked']),
                          feedback_callback=feedback)
        if 'fork' in which:
            server.insert(fork_marker(), feedback_callback=feedback)
        if 'tray' in which:
            server.insert(tray_marker(), feedback_callback=feedback)
        for n in which:
            menu.apply(server, n)
        server.applyChanges()

    def feedback(fb):
        n, pose = fb.marker_name, fb.pose
        ev = fb.event_type
        if ev not in (InteractiveMarkerFeedback.POSE_UPDATE, InteractiveMarkerFeedback.MOUSE_UP):
            return
        locked = L.v['locked']
        if locked and n in ('desk', 'shelf', 'fork'):
            return
        if ev == InteractiveMarkerFeedback.POSE_UPDATE:
            if n == 'robot' and locked:          # across the desk: the rest moves the other way
                ux, uy = L.desk_y_axis()
                d = pose.position.x * ux + pose.position.y * uy
                L.shift_robot(d - state.get('rd', 0.0))
                state['rd'] = d
                publish(('desk', 'bot', 'shelf', 'fork', 'tray'))
            elif n == 'tray':
                L.set_tray(pose.position.x, pose.position.y, yaw_of(pose.orientation))
            elif n == 'robot':                   # everything else turns about the base
                L.turn_world(yaw_of(pose.orientation) + L.v['robot_yaw'])
                publish(('desk', 'bot', 'shelf', 'fork'))
            elif n == 'desk':
                L.set_desk_pose(pose.position.x, pose.position.y, yaw_of(pose.orientation))
            elif n == 'bot':
                L.v['bot'] = [pose.position.x, pose.position.y, yaw_of(pose.orientation)]
            elif n == 'shelf':
                L.set_plate_centre(pose.position.x, pose.position.y, yaw_of(pose.orientation))
                publish(('fork', 'tray'))        # the fork (+ tray) follows the shelf
            elif n == 'fork':
                L.set_fork(pose.position.x, pose.position.y, pose.position.z)
                publish(('tray',))
            update_scene()
        else:                                    # MOUSE_UP: clamp / labels, scene, files
            state['rd'] = 0.0
            publish({'shelf': ('shelf', 'fork', 'tray'), 'desk': ('desk', 'robot'),
                     'fork': ('fork', 'tray'),
                     'robot': ('robot', 'desk', 'bot', 'shelf', 'fork', 'tray')}.get(n, (n,)))
            update_scene(force=True)
            store()

    def setter(**kv):
        def cb(fb):
            L.v.update({k: (v(L) if callable(v) else v) for k, v in kv.items()})
            publish()
            update_scene(force=True)
            store()
        return cb

    def blocked(n):
        if L.v['locked'] and n in ('robot', 'desk', 'shelf', 'fork'):
            log.warn(f'{n}: desk + shelf are locked (right click > unlock first)')
            return True
        return False

    def snap(fb):
        n = fb.marker_name
        if blocked(n):
            return
        if n == 'tray':
            L.v['tray_dyaw'] = 0.0
        elif n == 'bot':
            L.v['bot'][2] = round(L.v['bot'][2] / 90) * 90.0
        elif n in ('shelf', 'fork'):
            x, y = L.plate_centre()
            L.set_plate_centre(x, y, round(L.v['shelf'][2] / 90) * 90.0)
        elif n == 'desk':
            x, y, yd = L.desk_pose()
            L.set_desk_pose(x, y, round(yd / 90) * 90.0)
        elif n == 'robot':
            # desk direction (-robot_yaw) to the nearest 90 deg, the rest turning along
            L.turn_world(round(-L.v['robot_yaw'] / 90) * 90.0 + L.v['robot_yaw'])
        publish()
        update_scene(force=True)
        store()

    def reset(fb):
        n = fb.marker_name
        d = L.defaults()
        if blocked(n):
            return
        if n == 'tray':
            L.v.update(tray_dx=0.0, tray_dy=0.0, tray_dyaw=0.0)
        elif n == 'robot':
            L.turn_world(L.v['robot_yaw'] - d['robot_yaw'])
        elif n == 'desk':
            L.v.update(desk_x=d['desk_x'], edge_y=d['edge_y'], robot_yaw=d['robot_yaw'])
        elif n == 'bot':
            L.v['bot'] = list(d['bot']) if d['bot'] else L.default_bot()
        else:
            for k in ('shelf', 'shelf_x', 'fork_top', 'fork_y', 'tray_yaw_off'):
                L.v[k] = d[k]
        publish()
        update_scene(force=True)
        store()

    def guarded(cb):
        def f(fb):
            if not blocked('shelf'):
                cb(fb)
        return f

    menu = MenuHandler()
    menu.insert('lock / unlock desk + shelf (robot y, tray on D11, TurtleBot stay movable)',
                callback=setter(locked=lambda L_: not L_.v['locked']))
    hm = menu.insert('D11 height')
    lo, hi = ps.FORK_TOP[0] / 1000, ps.FORK_TOP[1] / 1000
    menu.insert('desk top height (arm reaches ~ here)', parent=hm,
                callback=guarded(setter(fork_top=lambda L_: clamp(L_.base.desk_height, lo, hi))))
    menu.insert(f'lowest {lo * 1000:.0f} mm over the floor', parent=hm,
                callback=guarded(setter(fork_top=lo)))
    menu.insert(f'highest {hi * 1000:.0f} mm over the floor', parent=hm,
                callback=guarded(setter(fork_top=hi)))
    tm = menu.insert('tray on D11: handle L->R')
    menu.insert('along the shelf X', parent=tm, callback=setter(tray_yaw_off=0.0))
    menu.insert('along the shelf Y (fork out)', parent=tm, callback=setter(tray_yaw_off=90.0))
    menu.insert('show tray on D11 (blocks the grasp plan)',
                callback=setter(show_tray=lambda L_: not L_.v['show_tray']))
    # ---- arm (simulation): straight down over D11 / the TurtleBot box
    import threading

    import numpy as np
    from sensor_msgs.msg import JointState

    from mycobot_calibration.geometry import make_T, rot_z
    from mycobot_calibration.goto_lookdown import driver_connection
    from mycobot_calibration.handeye_node import ARM_JOINTS
    from mycobot_calibration.moveit_client import MoveItClient
    from mycobot_tasks.red_cube_hover import DOWN, SEEDS_J23, TCP
    mv = MoveItClient(node, base_frame=args.frame)
    js = {}
    node.create_subscription(JointState, 'joint_states',
                             lambda m: js.update(zip(m.name, m.position)), 10)
    busy = threading.Lock()
    SEEDS = [(j2, j3, 0.0) for j2, j3 in SEEDS_J23] + [
        (-0.8, -1.2, 0.4), (0.3, -1.8, 0.0), (-1.2, -0.6, 0.3), (0.6, -2.0, -0.2),
        (-1.2, 0.4, -0.7)]

    def q_now():
        return np.array([js[n] for n in ARM_JOINTS]) if all(n in js for n in ARM_JOINTS) \
            else None

    def ik_over(xy, handle_z, tray_yaw):
        """Tool down over xy, fingers across the handle (tray yaw + 90), J1 toward it:
        the solution closest to now, the highest of --above .. 2 cm. No normal solution at
        any height: the reversed arm (J1 ~180 deg off, J2..J4 mirrored; e.g. D11 beside the
        rack, where the upper arm hits the rack), as tray_handle_pick (--no-flip = off).
        -> (q, tip z, reversed) or None."""
        q0 = q_now()
        for flip in ((False,) if args.no_flip else (False, True)):
            j1 = math.remainder(math.atan2(xy[1], xy[0]) + (math.pi if flip else 0.0),
                                2 * math.pi)
            m = -1.0 if flip else 1.0
            h = args.above
            while h >= 0.02 - 1e-9:
                p_ = np.array([xy[0], xy[1], handle_z + h])
                best = None
                for k in (0, 1):
                    T = make_T(rot_z(tray_yaw + math.pi / 2 + k * math.pi) @ DOWN, p_)
                    for a2, a3, a4 in SEEDS:
                        q = mv.compute_ik(T, TCP, ARM_JOINTS,
                                          np.array([j1, m * a2, m * a3, m * a4, 0, 0]),
                                          timeout=0.05)
                        if q is None:
                            continue
                        q = np.asarray(q)
                        if abs(math.remainder(q[0] - j1, 2 * math.pi)) > math.radians(60):
                            continue
                        c = 0.0 if q0 is None else float(np.sum((q - q0) ** 2))
                        if best is None or c < best[0]:
                            best = (c, q)
                if best is not None:
                    return best[1], h, flip
                h -= 0.02
        return None

    def arm_go(label, target):
        def run():
            if not busy.acquire(blocking=False):
                log.warn('arm: still moving')
                return
            try:
                conn = driver_connection(node)
                if conn != 'fake' and not args.real:
                    log.error(f'arm: driver connection is {conn!r}, not fake (simulation): '
                              'not moving (layout --real to allow)')
                    return
                if target is None:
                    q, msg = np.zeros(len(ARM_JOINTS)), 'all joints 0'
                else:
                    xy, hz, ty = target()
                    r = ik_over(xy, hz, ty)
                    if r is None:
                        log.error(f'arm: {label} ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm, r '
                                  f'{math.hypot(*xy) * 100:.0f} cm: no IK from '
                                  f'{args.above * 100:.0f} down to 2 cm over the handle top '
                                  '(out of reach, or a collision): move it closer / lower')
                        return
                    q, h, flip = r
                    msg = (f'finger tips {h * 100:.0f} cm over the handle top '
                           f'(z {(hz + h) * 1000:+.0f} mm), r {math.hypot(*xy) * 100:.0f} cm'
                           + (f', reversed arm (J1 {math.degrees(q[0]):+.0f} deg)' if flip
                              else ''))
                log.info(f'arm: going {label}: {msg}')
                for i in range(3):           # OMPL is random: a path that grazed the shelf
                    ok, why = mv.move_joints(ARM_JOINTS, q, velocity_scaling=0.3)
                    if ok:
                        break
                    log.warn(f'arm: {label}: try {i + 1} failed ({why})')
                log.info(f'arm: {label} ' + ('reached' if ok else
                                             f'FAILED: {why} (go to the start pose first?)'))
            finally:
                busy.release()
        return lambda fb: threading.Thread(target=run, daemon=True).start()

    def d11_target():
        x, y, z, yaw = L.tray_pose()
        return (x, y), z + args.tray_height, yaw

    def bot_target():
        bx, by, byaw = L.v['bot']
        return (bx, by), L.box_floor_z() + args.tray_height, math.radians(byaw)

    am = menu.insert('arm (simulation)')
    menu.insert('go above D11', parent=am, callback=arm_go('above D11', d11_target))
    menu.insert('go above the TurtleBot box', parent=am,
                callback=arm_go('above the TurtleBot box', bot_target))
    menu.insert('back to the start pose (all joints 0)', parent=am,
                callback=arm_go('to the start pose', None))
    # saved layouts (--presets): save the current one under the time, load one
    lm = menu.insert('layouts')

    def load_cb(name):
        def cb(fb):
            try:
                L.load_file(preset_path(name, args.presets))
            except (OSError, yaml.YAMLError) as e:
                log.error(f'layout {name}: {e}')
                return
            log.info(f'layout {name} loaded')
            publish()
            update_scene(force=True)
            store()
        return cb

    def save_cb(fb):
        name = time.strftime('%Y%m%d-%H%M%S')
        log.info(f'layout saved as {L.save_as(name, args.presets)} (rename the file as you '
                 'like; load it: layouts > load, or --load NAME)')
        menu.insert(f'load: {name}', parent=lm, callback=load_cb(name))
        publish()
    menu.insert('save as <date-time>', parent=lm, callback=save_cb)
    for n in presets(args.presets)[:20]:
        menu.insert(f'load: {n}', parent=lm, callback=load_cb(n))
    menu.insert('snap yaw to 90 deg', callback=snap)
    menu.insert('reset this one', callback=reset)

    def wait_scene():
        if cli.service_is_ready():
            update_scene(force=True)
            remove(['calibration_table'])
            scene_timer.cancel()
            log.info('planning scene: layout applied (calibration_table removed)')

    publish()
    store()
    node.create_timer(0.2, flush)
    scene_timer = node.create_timer(1.0, wait_scene)
    log.info(f'layout on /layout (RViz InteractiveMarkers), saved to {L.path}; '
             'right click a handle for presets')
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor()               # the arm menu waits on MoveIt in a thread
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
