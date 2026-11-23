"""Search regions: the shelf ('shelf', for now the desk top beside the robot: trays are
picked from / placed on it, no marker there) and the TurtleBot ('bot').
x / y are dragged in RViz (interactive markers); the heights are picked from the right-click
menu or typed on the command line. Saved to ~/.ros/regions.yaml; tray_handle_pick reads that
file each time it scans (no restart needed).

  ros2 run mycobot_tasks regions                 # RViz: Interact tool (i), display
                                                 # InteractiveMarkers, namespace /regions
  ros2 run mycobot_tasks regions --print         # show the saved regions, no ROS
  ros2 run mycobot_tasks regions --set bot --z -0.111 --view 0.35     # numbers, no ROS
  ros2 run mycobot_tasks regions --set shelf --x -0.15,0.05 --y 0.14,0.27
  ros2 run mycobot_tasks regions --set bot --center 0.30,-0.05     # move, size kept
  ros2 run mycobot_tasks regions --set bot --shift 0.02,0            # move by dx,dy

Per region:
  x, y   the box corners: drag sphere A / B in RViz (in the plane of the surface);
         drag the box itself to move the whole region (size kept)
  z      the surface the tray / the TurtleBot marker is on (base frame) [m]
  view   camera height over that surface when scanning / measuring [m]
Right click a sphere: 'surface z' and 'camera height' presets, 'reset this region'.
A running `regions` node picks up --set changes from the file within a second.
"""
import argparse
import copy
import os
import sys

import yaml

PATH = os.path.expanduser('~/.ros/regions.yaml')
VIEW = 0.28                       # default camera height over the surface [m]
# defaults (base frame, m): shelf = the desk top on the +y side of the robot (stand-in
# until the real shelf is built; desk y up to +0.29, front end x +0.06), TurtleBot in front
# of the desk below it; z = surface (tray bottom / marker plane)
DEFAULTS = {
    'shelf': dict(x=[-0.15, 0.05], y=[0.14, 0.27], z=-0.007, view=VIEW),
    'bot': dict(x=[-0.10, 0.20], y=[-0.60, -0.40], z=-0.280, view=VIEW),
}
COLOURS = {'shelf': (0.1, 0.8, 0.2), 'bot': (0.2, 0.4, 1.0)}
LABELS = {'shelf': 'shelf area', 'bot': 'TurtleBot area'}
# surface presets (base frame z): place_scene desk top -0.007, floor -0.306 (desk 0.299),
# TurtleBot box floor (floor + 0.026, box behind the Burger)
FLOOR = -0.306
Z_PRESETS = [('desk top', -0.007), ('TurtleBot box floor', FLOOR + 0.026),
             ('floor', FLOOR)]
VIEW_PRESETS = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40]


def _clean(v, default):
    return dict(x=sorted(float(t) for t in v.get('x', default['x'])),
                y=sorted(float(t) for t in v.get('y', default['y'])),
                z=float(v.get('z', default['z'])),
                view=float(v.get('view', default['view'])))


def load(path=PATH):
    """-> {name: dict(x=[x0, x1], y=[y0, y1], z, view)} (missing ones = defaults)."""
    out = copy.deepcopy(DEFAULTS)
    try:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        for k, v in d.items():
            if k in out and isinstance(v, dict):
                out[k] = _clean(v, DEFAULTS[k])
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        pass
    return out


def saved(name, path=PATH):
    """The region from the file, or None when the file / the region is not there."""
    try:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    return load(path)[name] if name in d else None


def save(regions, path=PATH):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump({k: dict(x=[round(v, 4) for v in sorted(r['x'])],
                                y=[round(v, 4) for v in sorted(r['y'])],
                                z=round(r['z'], 4), view=round(r['view'], 4))
                        for k, r in regions.items()}, f, sort_keys=False)
    os.replace(tmp, path)


def describe(name, r):
    return (f"{name:5s} x {r['x'][0] * 1000:5.0f}..{r['x'][1] * 1000:5.0f}  "
            f"y {r['y'][0] * 1000:5.0f}..{r['y'][1] * 1000:5.0f}  z {r['z'] * 1000:5.0f} mm  "
            f"camera {r['view'] * 100:.0f} cm over it "
            f"({(r['x'][1] - r['x'][0]) * 100:.0f} x {(r['y'][1] - r['y'][0]) * 100:.0f} cm)")


def pair(t):
    v = [float(s) for s in t.split(',')]
    if len(v) != 2:
        raise argparse.ArgumentTypeError('two values: a,b')
    return sorted(v)


def pair2(t):
    v = [float(s) for s in t.split(',')]
    if len(v) != 2:
        raise argparse.ArgumentTypeError('two values: a,b')
    return v


def centre(r):
    return (r['x'][0] + r['x'][1]) / 2, (r['y'][0] + r['y'][1]) / 2


def move_to(r, cx, cy):
    """Region centre to (cx, cy), width / depth kept."""
    hx, hy = (r['x'][1] - r['x'][0]) / 2, (r['y'][1] - r['y'][0]) / 2
    r['x'], r['y'] = [cx - hx, cx + hx], [cy - hy, cy + hy]


def main():
    a = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    a.add_argument('--file', default=PATH)
    a.add_argument('--print', action='store_true', help='print the saved regions only')
    a.add_argument('--set', choices=sorted(DEFAULTS), help='change this region, then exit')
    a.add_argument('--x', type=pair, help='--set: x range x0,x1 [m]')
    a.add_argument('--y', type=pair, help='--set: y range y0,y1 [m]')
    a.add_argument('--z', type=float, help='--set: surface z (base frame) [m]')
    a.add_argument('--view', type=float, help='--set: camera height over the surface [m]')
    a.add_argument('--center', type=pair2,
                   help='--set: move the region centre to cx,cy, size kept [m]')
    a.add_argument('--shift', type=pair2, help='--set: move the region by dx,dy, size kept [m]')
    a.add_argument('--frame', default='base')
    argv, i = [], 0
    raw = sys.argv[1:]
    while i < len(raw):                   # '--y -0.6,-0.4' / '--z -0.1': negative values
        if raw[i] in ('--x', '--y', '--z', '--center', '--shift') and i + 1 < len(raw):
            argv.append(f'{raw[i]}={raw[i + 1]}')
            i += 2
        else:
            argv.append(raw[i])
            i += 1
    args, _ = a.parse_known_args(argv)
    regions = load(args.file)
    if args.set:
        r = regions[args.set]
        for k in ('x', 'y', 'z', 'view'):
            if getattr(args, k) is not None:
                r[k] = getattr(args, k)
        if args.center is not None:
            move_to(r, *args.center)
        if args.shift is not None:
            move_to(r, *(c + d for c, d in zip(centre(r), args.shift)))
        save(regions, args.file)
        print('saved ' + describe(args.set, r))
        return
    if args.print:
        for k, r in regions.items():
            print(describe(k, r))
        return

    import rclpy
    from geometry_msgs.msg import Point
    from interactive_markers import InteractiveMarkerServer, MenuHandler
    from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl,
                                        InteractiveMarkerFeedback, Marker)

    rclpy.init()
    node = rclpy.create_node('regions')
    server = InteractiveMarkerServer(node, 'regions')
    menu = MenuHandler()
    mtime = [None]

    def box_marker(name):
        r = regions[name]
        im = InteractiveMarker()
        im.header.frame_id = args.frame
        im.name = f'{name}_box'
        im.description = (f"{LABELS[name]}  z {r['z'] * 1000:.0f} mm, camera "
                          f"{r['view'] * 100:.0f} cm")
        im.pose.position.x = (r['x'][0] + r['x'][1]) / 2
        im.pose.position.y = (r['y'][0] + r['y'][1]) / 2
        im.pose.position.z = r['z'] + 0.005
        im.pose.orientation.w = 1.0
        im.scale = 0.1
        # drag the box: the whole region moves in x / y, size kept
        c = InteractiveMarkerControl(always_visible=True, name='move_xy',
                                     interaction_mode=InteractiveMarkerControl.MOVE_PLANE)
        c.orientation.w = c.orientation.y = 0.7071067811865476   # plane normal = base z
        m = Marker(type=Marker.CUBE)
        m.scale.x = max(0.005, r['x'][1] - r['x'][0])
        m.scale.y = max(0.005, r['y'][1] - r['y'][0])
        m.scale.z = 0.01
        m.color.r, m.color.g, m.color.b = COLOURS[name]
        m.color.a = 0.35
        c.markers.append(m)
        cam = Marker(type=Marker.ARROW)      # camera height: arrow from there down
        cam.scale.x, cam.scale.y, cam.scale.z = 0.006, 0.012, 0.02
        cam.color.r, cam.color.g, cam.color.b = COLOURS[name]
        cam.color.a = 0.8
        cam.points = [Point(x=0.0, y=0.0, z=r['view']), Point(x=0.0, y=0.0, z=0.01)]
        c.markers.append(cam)
        im.controls.append(c)
        return im

    def handle_marker(name, which):
        r = regions[name]
        im = InteractiveMarker()
        im.header.frame_id = args.frame
        im.name = f'{name}_{which}'
        im.description = f"{LABELS[name]} {which}"
        im.pose.position.x = r['x'][0] if which == 'A' else r['x'][1]
        im.pose.position.y = r['y'][0] if which == 'A' else r['y'][1]
        im.pose.position.z = r['z'] + 0.005
        im.pose.orientation.w = 1.0
        im.scale = 0.08
        s = Marker(type=Marker.SPHERE)
        s.scale.x = s.scale.y = s.scale.z = 0.025
        s.color.r, s.color.g, s.color.b = COLOURS[name]
        s.color.a = 0.9
        c = InteractiveMarkerControl(always_visible=True, name='move_xy',
                                     interaction_mode=InteractiveMarkerControl.MOVE_PLANE)
        c.orientation.w = c.orientation.y = 0.7071067811865476   # plane normal = base z
        c.markers.append(s)
        im.controls.append(c)
        return im

    def publish(name, handles=True, box=True):
        if box:
            server.insert(box_marker(name), feedback_callback=feedback)
            menu.apply(server, f'{name}_box')
        if handles:
            for w in 'AB':
                server.insert(handle_marker(name, w), feedback_callback=feedback)
                menu.apply(server, f'{name}_{w}')
        server.applyChanges()

    def store(name):
        save(regions, args.file)
        mtime[0] = os.path.getmtime(args.file)
        node.get_logger().info('saved ' + describe(name, regions[name]))

    def feedback(fb):
        name, which = fb.marker_name.rsplit('_', 1)
        if name not in regions:
            return
        if which == 'box':
            if fb.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
                move_to(regions[name], fb.pose.position.x, fb.pose.position.y)
                publish(name, box=False)          # the corner spheres follow
            elif fb.event_type == InteractiveMarkerFeedback.MOUSE_UP:
                publish(name)
                store(name)
            return
        if fb.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            r = regions[name]
            i = 0 if which == 'A' else 1
            r['x'][i], r['y'][i] = fb.pose.position.x, fb.pose.position.y
            publish(name, handles=False)
        elif fb.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            r = regions[name]
            r['x'].sort()
            r['y'].sort()
            store(name)

    def setter(key, value):
        def cb(fb):
            name = fb.marker_name.rsplit('_', 1)[0]
            regions[name][key] = value
            publish(name)
            store(name)
        return cb

    def reset(fb):
        name = fb.marker_name.rsplit('_', 1)[0]
        regions[name] = copy.deepcopy(DEFAULTS[name])
        publish(name)
        store(name)

    zm = menu.insert('surface z')
    for label, z in Z_PRESETS:
        menu.insert(f'{label}  ({z * 1000:.0f} mm)', parent=zm, callback=setter('z', z))
    vm = menu.insert('camera height')
    for v in VIEW_PRESETS:
        menu.insert(f'{v * 100:.0f} cm over the surface', parent=vm, callback=setter('view', v))
    menu.insert('reset this region', callback=reset)

    def reload():                         # `regions --set ...` while this node runs
        try:
            t = os.path.getmtime(args.file)
        except OSError:
            return
        if mtime[0] is None:
            mtime[0] = t
        elif t != mtime[0]:
            mtime[0] = t
            regions.update(load(args.file))
            for k in regions:
                publish(k)
            node.get_logger().info('reloaded ' + args.file)

    for k in regions:
        publish(k)
        node.get_logger().info(describe(k, regions[k]))
    node.create_timer(1.0, reload)
    node.get_logger().info(f'regions on /regions (RViz InteractiveMarkers), saved to {args.file}'
                           '; heights: right click a sphere, or regions --set')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
