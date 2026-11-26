"""Command line: options -> HandleApp, window, --do first action."""
import argparse
import os
import threading

import cv2

import rclpy
from rclpy.executors import MultiThreadedExecutor

from mycobot_tasks.red_cube_hover import Cube
from mycobot_tasks.handle_pick.app import HandleApp
from mycobot_tasks.handle_pick.base import _tee_stdout
from mycobot_tasks.handle_pick.window import view


CONFIG = 'tray_handle_pick.yaml'      # share/mycobot_tasks/config: defaults by purpose
GROUPS = {   # config section -> --help title
    'run': 'per run (command line only)',
    'tray': 'tray size / grasp heights (--tray-type presets: tray_types)',
    'markers': 'handle ArUco markers',
    'detection': 'detection: camera frames, looking again (tracking)',
    'search': 'search pose / region scan (regions.yaml)',
    'turtlebot': 'TurtleBot marker / box / place range',
    'gripper': 'gripper',
    'scene': 'planning scene (place_scene)',
    'moveit': 'MoveIt moves (camera poses etc., outside the relay paths)',
    'relay_plan': 'relay planning: path search',
    'relay_safety': 'relay clearances (collision checks)',
    'relay_speed': 'relay speeds',
    'relay_end': 'after the relay',
    'files': 'files / topics',
}


def _read_yaml(path):
    import yaml
    with open(os.path.expanduser(path)) as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise SystemExit(f'{path}: not a mapping of sections')
    return d


def load_config(user=None):
    """Package defaults (config/tray_handle_pick.yaml) + the --config file over them.
    -> (flat {dest: value}, tray_types, keys set by the --config file)."""
    from ament_index_python.packages import get_package_share_directory
    base = _read_yaml(os.path.join(get_package_share_directory('mycobot_tasks'), 'config',
                                   CONFIG))
    flat, where = {}, {}
    for sec, vals in base.items():
        if sec == 'tray_types':
            continue
        for k, v in (vals or {}).items():
            flat[k], where[k] = v, sec
    types = dict(base.get('tray_types') or {})
    mine = set()
    if user:
        for sec, vals in _read_yaml(user).items():
            if sec == 'tray_types':
                types.update(vals or {})
                continue
            for k, v in (vals or {}).items():
                if k not in flat:
                    raise SystemExit(f'--config {user}: unknown key {sec}.{k}')
                if where[k] != sec:
                    raise SystemExit(f'--config {user}: {k} belongs under {where[k]}:, not {sec}:')
                flat[k] = v
                mine.add(k)
    return flat, types, mine


def main():
    from ament_index_python.packages import get_package_share_directory
    from mycobot_tasks import tray_handle_pick
    import sys
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    user_cfg = pre.parse_known_args(sys.argv[1:])[0].config
    cfg, tray_types, cfg_mine = load_config(user_cfg)
    a = argparse.ArgumentParser(
        description=tray_handle_pick.__doc__.split('\n\n')[0],
        epilog=f'defaults: share/mycobot_tasks/config/{CONFIG} (sections = the groups above)')
    grp = {k: a.add_argument_group(f'{k}: {t}') for k, t in GROUPS.items()}
    # ---- run: per run (command line only)
    G = grp['run'].add_argument
    G('--go', action='store_true', help='really move (else plan only)')
    G('--real', action='store_true', help='allow moving a real (non-fake) robot')
    G('--auto', action='store_true', help='no step questions')
    G('--no-show', action='store_true', help='no camera window')
    G('--config', default=None,
      help=f'YAML over the defaults ({CONFIG}): only the keys to change, under the same '
           'section names')
    G('--do', choices=['', 'none', 'detect', 'd11bot', 'botd11', 'deskbot', 'botdesk'],
      default='', help='first action: detect (d), d11bot (n), botd11 (j), deskbot (e), '
                       'botdesk (r)')
    G('--search', action='store_true', help='go to the search pose first (it becomes home)')
    G('--replay', nargs='?', const='~/.ros/tray_handle_last.yaml', default=None,
      help='SIMULATION: use the saved detection instead of the camera and start from the '
           'arm pose of that detection (run with handeye.launch.py sim:=true)')
    G('--layout-poses', action='store_true',
      help='n / j, SIMULATION: D11 tray / TurtleBot / tray in the box as arranged in RViz '
           '(ros2 run mycobot_tasks layout files, read at every n / j), no camera')
    G('--relay-desk', type=lambda t: [float(v) for v in t.split(',')], default=None,
      help='e / r (desk <-> TurtleBot): x,y,yaw: --relay-deskthe tray on the desk, handle midpoint [m] '
           'and L -> R [deg], base frame. r: where it is put (default: the planner picks a '
           'desk spot); e with --layout-poses (simulation, no camera): where it stands '
           '(default: a desk spot the planner picks); e on the real robot: measured')
    G('--relay-spots', type=lambda t: None if t.strip().lower() == 'auto' else
      [tuple(float(v) for v in p.split(',')) for p in t.split(';') if p.strip()],
      default=None,
      help='n / j / e / r: desk spot(s) for the handle midpoint, "x,y;x,y;..." base [m] (the '
           'tray must lie fully on the desk), e.g. -0.22,0 behind the arm base; default '
           '(auto): a desk grid at --relay-spot-r. e looks for the tray only there (camera '
           'over each spot; auto: the first one), r puts it there')
    G('--relay-spots-bot', type=lambda t: None if t.strip().lower() == 'auto' else
      [tuple(float(v) for v in p.split(',')) for p in t.split(';') if p.strip()],
      help='e / r only (desk <-> TurtleBot): desk spot(s) "x,y;x,y;..." base [m], tried in '
           'turn, unless --relay-spots / --relay-desk is given; auto = like n / j '
           '(--relay-spot-r). Default: right of the arm base, so the held tray does not '
           'pass behind it (11-26: the tray hit the base back on the way to (-0.20,-0.08))')
    G('--sim-speed', type=float, default=1.0,
      help='SIMULATION only (fake driver): motions this many times faster (relay paths '
           'and MoveIt moves; e.g. 3 = 3x, 0.5 = half speed). The real robot ignores it')
    # ---- tray: 트레이 치수 / 잡는 높이 (--tray-type 프리셋이 덮어씀: tray_types)
    G = grp['tray'].add_argument
    G('--tray-type', choices=list(tray_types),
      help='tray preset (config tray_types; --config / command-line values win): ' + '; '.join(
          f'{k}: ' + ' '.join(f'--{o.replace("_", "-")} {v}' for o, v in d.items())
          for k, d in tray_types.items()))
    G('--tray-height', type=float,
      help='tray bottom -> handle top (marker plane) [m]: the place heights come from it')
    G('--tray-height-tol', type=float,
      help='warn when the measured handle top over --table-z differs more [m]')
    G('--relay-tray', type=lambda t: [float(v) for v in t.split(',')],
      help='n / j: held tray body for the collision checks: length along the finger '
           'closing direction, width along the handle bar, height [m] (shelf tray: guide '
           'part E 150 x 100 x 56)')
    G('--grasp-depth', type=float,
      help='finger tips this far below the handle top (marker plane) when grasping [m]')
    G('--approach-above', type=float,
      help='approach / lift: finger tips this far above the handle top [m]')
    G('--servo-above', type=float,
      help='first go with the finger tips this far above the handle top [m]')
    G('--grip-yaw-offset', type=float,
      help='TCP yaw relative to the L -> R line [deg]; fingers close along the TCP x '
           'axis, so 90 = across the bar')
    G('--max-handle-h', type=float,
      help='handle top higher than this over --table-z = wrong --table-z [m]')
    # ---- markers: 손잡이 ArUco 마커
    G = grp['markers'].add_argument
    G('--marker-size', type=float, help='black marker side [m]')
    G('--gap', type=float,
      help='handle length = gap between the inner edges of the two cut marker TILES '
           '(white margin included) [m]; centre spacing checked = gap + --tile-size '
           '(0 = no check)')
    G('--tile-size', type=float, help='cut marker tile incl. the white margin [m] (printed 30 mm)')
    G('--spacing', type=float,
      help='or the L-R marker centre spacing directly [m] (overrides --gap)')
    G('--max-spacing-err', type=float, help='refuse above this [m]')
    G('--max-tilt', type=float, help='marker plane tilt limit [deg]')
    # ---- detection: 감지: 카메라 프레임, 다시 보기(tracking)
    G = grp['detection'].add_argument
    G('--frames', type=int, help='camera frames per detection')
    G('--pin-timeout', type=float, help='wait for them [s]')
    G('--max-spread', type=float, help='midpoint xy jitter limit [m]')
    G('--no-track', dest='track', action='store_false',
      help="detection: don't move the camera (one marker / far / off-centre / at an angle)")
    G('--recenter', type=int, help='camera moves per detection at most')
    G('--min-marker-px', type=float,
      help='markers smaller than this [px side] = too far: look from closer')
    G('--max-off-centre', type=float,
      help='midpoint further from the image centre (fraction of the width) = recentre')
    G('--max-view', type=float,
      help='camera axis vs marker normal above this [deg] = look from straight above')
    G('--track-tol', type=float,
      help='estimate within this of the last look point = no further camera move [m]')
    G('--scan-step', type=float,
      help='nothing seen: look this far +-x / +-y around the search point [m]')
    G('--look-max-r', type=float, help='camera look point at most this far from the base [m]')
    # ---- search: 탐색 자세 / 영역 스캔 (regions.yaml)
    G = grp['search'].add_argument
    G('--search-x', type=float)
    G('--search-y', type=float)
    G('--search-height', type=float)
    G('--search-min-height', type=float)
    G('--regions',
      help='search regions (ros2 run mycobot_tasks regions, set in RViz): shelf = where d / n '
           'look when nothing is seen, bot = where the TurtleBot / the tray on it is looked for')
    G('--scan-height', type=float,
      help="region scan: camera this far over the region surface [m] (lower if unreachable); "
           "default: the region's camera height (regions.yaml, 0.28)")
    G('--scan-pull', type=float,
      help='region scan: a view out of reach straight down is taken obliquely from up to '
           'this much closer to the base [m] (0 = straight down only)')
    G('--no-last', dest='use_last', action='store_false',
      help="nothing seen: don't look at the last tray / TurtleBot position before the "
           'region scan')
    G('--chase', type=int,
      help='scan: a dark patch cut by the image border (a marker partly outside the image) '
           'is followed this many camera moves (0 = off)')
    G('--chase-dark', type=float,
      help='chase: pixels darker than this (0..255, and the Otsu level) count as marker black')
    G('--scan-overlap', type=float,
      help='region scan: neighbouring views overlap by this fraction of the footprint')
    # ---- turtlebot: TurtleBot 마커 / 박스 / 놓는 범위
    G = grp['turtlebot'].add_argument
    G('--bot-id', type=int, help='TurtleBot ArUco ID (DICT_4X4_50)')
    G('--bot-marker-size', type=float, help='TurtleBot marker side [m]')
    G('--bot-heading-id', type=int,
      help='Burger heading marker ID (DICT_4X4_50, on the back of the LiDAR): r / j take '
           'the Burger direction from it every time (tray L->R alone is +-180); -1 = off '
           '(Burger assumed facing base +x)')
    G('--bot-heading-size', type=float, help='Burger heading marker side (black square) [m]')
    G('--heading-keep-s', type=float,
      help='heading marker not in a view: the one seen up to this long ago is used [s]')
    G('--bot-marker-dz', type=float,
      help='TurtleBot marker plane this far ABOVE the surface the tray goes on [m] '
           '(0 = marker lies on that surface, e.g. on the box floor)')
    G('--bot-place-offset', type=lambda t: [float(v) for v in t.split(',')],
      help='place spot (handle midpoint) from the marker centre, dx,dy[,dz] [m]: dx,dy along '
           'the marker x / y (--bot-offset-frame marker, ArUco x = to the right of the '
           'printed marker, y = up) or base x / y (base); dz raises (+) / lowers (-) the '
           'place height')
    G('--bot-offset-frame', choices=['marker', 'base'], help='frame of --bot-place-offset dx,dy')
    G('--bot-yaw-offset', type=float,
      help="placing in the box: tray L -> R = marker +x turned by this [deg] (90: the shelf "
           "tray's 15 cm side along the box 20 cm side; 0 = the 16 x 10 test tray)")
    G('--bot-box-offset', type=lambda t: [float(v) for v in t.split(',')],
      help='scene: box centre from the marker centre, dx,dy in the marker frame [m]')
    G('--bot-keepout-h', type=float,
      help='while the TurtleBot pose is not known (finding it / the tray on it): its whole '
           'area is an obstacle this high over the floor (Burger 0.192 with the LiDAR on top '
           '+ 3 cm) and camera views keep the gripper tip over it [m]')
    G('--bot-keepout-margin', type=float,
      help='... the TurtleBot area (regions.yaml) grown by this on each side [m]')
    G('--no-bot-scene', dest='bot_scene', action='store_false',
      help="don't put the TurtleBot + box into the planning scene where the marker was found")
    G('--bot-rim', type=float,
      help='TurtleBot box rim over the surface the tray goes on (marker = box floor) [m]: '
           'carrying in and scanning while holding keep the tray bottom --carry-clear over it')
    G('--carry-clear', type=float,
      help='scanning while holding: the tray bottom this far over the box rim [m]')
    G('--bot-close-view', type=float,
      help='r / j: the tray in the TurtleBot box measured again with the camera this high over '
           'it, once the Burger is in the scene (the search view is high: small markers, '
           'poor tilt) [m] (0 = off)')
    G('--bot-r-range', type=lambda t: [float(v) for v in t.split(',')],
      help='n / j / e / r: the TurtleBot box centre (marker / tray in it) must be this far from '
           'the arm base axis, min,max [m]; outside: ERROR at once, no planning (sim: 30.5 and '
           '31.1 cm fail, 22 cm fails)')
    G('--bot-place-r', type=float,
      help='n / e (placing): the tray handle put no farther than this from the arm base (inside '
           'the --bot-r-range max, so r / j find it within range despite the marker error) [m]')
    G('--bot-place-wall', type=float,
      help='n / e: box farther than --bot-place-r: the tray moved toward the arm base inside '
           'the box as far as this gap to its walls stays (any tray size / box direction) [m]')
    G('--relay-place-shift', type=lambda t: [float(v) for v in t.split(',')],
      help='n / e: no way down into the TurtleBot box at its centre -> the tray put this far '
           'off it, each value in turn (toward the arm base first) [m] (0 = centre only)')
    # ---- gripper: 그리퍼
    G = grp['gripper'].add_argument
    G('--open-position', type=float, help='gripper open command (1.0 = 10 cm)')
    G('--grasp-open', type=float,
      help='opening around the handle: going down to grasp it and releasing it (0.5 = 5 cm; '
           'fully open fingers reach the TurtleBot beside its box). Detection still opens '
           '--open-position (fingers out of the camera view)')
    G('--hold-grip', type=float,
      help='gripper opening the legs with the tray held are planned with (collision check; '
           '0 = closed)')
    G('--grip-settle', type=float,
      help='after closing wait this long, then read the finger gap [s]')
    G('--min-grip-gap', type=float, help='closed gripper gap below this = nothing grasped [mm]')
    G('--full-open-mm', type=float, help='gripper gap at joint value 1.0 [mm]')
    G('--place-drop', type=float,
      help='putting down (TurtleBot box): open this far above the grasp height [m]')
    # ---- scene: 계획 장면 (place_scene)
    G = grp['scene'].add_argument
    G('--no-scene', dest='scene', action='store_false',
      help="don't load place_scene; use the old 1.5 x 1.5 m table box at --table-z instead")
    G('--scene-args',
      help='place_scene options as one string, e.g. "--with-bot --bot 0.02,-0.40" or '
           '"--edge-points 0.05,-0.262,0.25,-0.245"')
    G('--table-z', type=float,
      help='z of the surface the tray stands on, base frame [m] (default: the shelf region z '
           'from regions.yaml, else the desk top -0.007; below the desk e.g. the floor -0.306 '
           'or a shelf fork)')
    G('--table-margin', type=float, help='desk collision box raised by this [m]')
    # ---- moveit: MoveIt 이동 (카메라 자세 등, relay 경로 밖)
    G = grp['moveit'].add_argument
    G('--planner', choices=['pilz', 'ompl'],
      help='pilz: straight lines (LIN) for every move near / with the tray, PTP otherwise; '
           'ompl: the old any-path planning')
    G('--velocity', type=float)
    G('--accel', type=float)
    G('--slow-velocity', type=float, help='near the tray: descent and lift')
    G('--lin-velocity', type=float,
      help='LIN moves: scaling of pilz_cartesian_limits (max_trans_vel 0.10 m/s)')
    G('--lin-accel', type=float,
      help='LIN moves: acceleration scaling (Pilz checks the joint limits along the line)')
    G('--mid-settle', choices=['precise', 'once', 'fast'],
      help='intermediate motions (lift, up, carry, back, home, over the handle / put spot): '
           'driver settle_mode. fast = no correction, done within the driver '
           'fast_tolerance_deg; once = at most 1 correction; precise (default) = as before, '
           'like the grasp / put down / camera views (settle + up to max_corrections)')
    G('--driver-node', help='driver node (settle_mode parameter)')
    # ---- relay_plan: relay 계획: 경로 탐색
    G = grp['relay_plan'].add_argument
    G('--relay-tilt', type=float,
      help='n / j: carrying, the tray may tilt this much [deg] (grasp / put down level)')
    G('--relay-tilt-weight', type=float,
      help='n / j: pull toward a level tray inside --relay-tilt (0 = any tilt up to the limit)')
    G('--relay-carry-z', type=float,
      help='n / j: finger tip height while carrying over the grid, base z [m]')
    G('--relay-pull-r', type=float,
      help='n / j: out of the shelf the tray is pulled toward the base to this radius [m]')
    G('--relay-grid', type=lambda t: [float(v) for v in t.split(',')],
      help='n / j: carry grid xmin,xmax,ymin,ymax [m] (2 cm steps)')
    G('--relay-min-r', type=float, help='n / j: grid points this close to '
      'the base axis are left out [m]')
    G('--relay-grid-budget', type=lambda t: [int(v) for v in t.split(',')],
      help='n / j / e / r: desk -> TurtleBot first tried quickly: turn at the desk spot only, '
           'each carry grid search stopped after this many points, one pass per value '
           '(e.g. 40,120); nothing -> everything without a cap (0 = no quick pass) [points]')
    G('--relay-spot-r', type=lambda t: [float(v) for v in t.split(',')],
      help='n / j: automatic desk spots: radius range from the base axis rmin,rmax [m]')
    G('--relay-max-spots', type=int,
      help='n / j: desk spots tried when desk -> TurtleBot does not work from one [count]')
    G('--relay-grasp-tries', type=int,
      help='n / j: different grasp poses (IK branches) tried on D11 when the pull out from '
           'one hits the rack / the arm itself [count]')
    G('--relay-plan-timeout', type=float,
      help='n / j / e / r: one planning part longer than this = failed (ERROR, the arm back '
           'home) [s] (0 = no limit). The real robot plans ~3x slower than the simulation')
    G('--relay-lift-max', type=float,
      help='n / j / e / r: empty-arm moves to a higher goal (home, start) first go straight up '
           'by up to this (tool orientation kept), then OMPL [m] (0 = off)')
    G('--max-step-jump', type=float,
      help='... and refused if one step moves any joint more than this [deg] (branch flip)')
    # ---- relay_safety: relay 안전 여유 (충돌 검사)
    G = grp['relay_safety'].add_argument
    G('--relay-plan-pad', type=lambda t: [float(v) for v in t.split(',')],
      help='n / j / e / r: plan with the held tray grown by this on every side [m] (clearance '
           'from the arm base / shelf), several = tried in turn until one works (most first; '
           'default 0.005,0.003,0: 0 mm used = WARNING)')
    G('--relay-base-pad', type=float,
      help='n / j / e / r: the held tray keeps at least this from the arm base, link1 and the '
           'robot_mount plates (on top of --relay-plan-pad, fixed: not relaxed) [m] (0 = off)')
    G('--relay-pad', type=float,
      help='n / j: after planning, every path is checked densely again, once as planned '
           '(contact = refused) and once with the tray grown by this on every side: the '
           'spots with less clearance are listed [m]')
    G('--relay-touch', type=float,
      help='n / j / e / r: tray bottom within this over the surface it is put on / taken from '
           '(D11 fork / desk top / TurtleBot box floor, not its walls): contacts with that '
           'surface are normal (not collisions) [m]; covers the shelf margin (1 cm) + the '
           'tray clearance (--relay-plan-pad)')
    G('--relay-desk-clear', type=float,
      help='n / j / e / r: empty-arm moves (OMPL: to the start, arm switch, home) keep this far '
           'above the desk top [m] (0 = off)')
    G('--relay-desk-clear-r', type=float,
      help='--relay-desk-clear: a square this half size around the arm base is left out [m]')
    # ---- relay_speed: relay 속도
    G = grp['relay_speed'].add_argument
    G('--relay-speed', type=float,
      help='n / j / e / r: fastest joint speed of the relay paths, empty arm [rad/s]')
    G('--relay-held-speed', type=float,
      help='n / j / e / r: the same while holding the tray (--relay-tcp-speed scaled alike) '
           '[rad/s] (0.25 tried on the real arm 11-25: no less error off the path)')
    G('--relay-tcp-speed', type=float, help='n / j: fastest finger tip / tray edge speed [m/s]')
    G('--relay-accel', type=float,
      help='n / j: speeding up / slowing down at most this [m/s2] (smooth start and stop of '
           'every path: what is in the tray must not slide)')
    G('--relay-lever', type=float,
      help='n / j: tool rotation counts as the motion of a point this far away (tray edge) [m]')
    G('--relay-home-speed', type=float,
      help='n / j / e / r: fastest joint speed of the empty way home after the hand-over '
           '(OMPL part; the straight-up part beside the tray stays at --relay-speed) [rad/s]')
    G('--relay-home-tcp-speed', type=float,
      help='n / j / e / r: the same, fastest finger tip speed (--relay-accel scaled alike) '
           '[m/s]')
    # ---- relay_end: relay 끝난 뒤
    G = grp['relay_end'].add_argument
    G('--relay-end', choices=['home', 'stay'],
      help='n / j / e / r: after the hand-over back to the home pose (--relay-home), then '
           'the gripper closed; stay = the arm stays where it ended')
    G('--relay-place-check', type=lambda t: t.strip().lower() in ('1', 'true', 'yes', 'on'),
      help='e / r / n (real robot): after putting the tray down, the camera measures it again '
           'and the error vs the plan is logged (--relay-check-log); true / false')
    G('--relay-check-warn', type=lambda t: [float(v) for v in t.split(',')],
      help='place check: WARNING above this position [m], yaw [deg] error')
    G('--relay-check-log', help="place check: CSV file, one row per check ('' = off)")
    G('--relay-home', type=lambda t: [float(v) for v in t.split(',')],
      help='n / j / e / r: the home pose they end in (and go to after an error before '
           'carrying), J1..J6 [deg]; empty = the arm pose when the program started (not '
           'moved by the camera re-centring of a detection)')
    # ---- files: 파일 / 토픽
    G = grp['files'].add_argument
    G('--relay-d11',
      help='n: where the camera looks for the tray on D11 first; j: where the tray goes on D11 '
           '(layout export: handle top xyz + L -> R yaw)')
    G('--layout-dir', help='--layout-poses: folder of the layout_*.yaml files (layout exports)')
    G('--relay-urdf',
      help="n / j: URDF file for the relay IK when robot_state_publisher does not answer")
    G('--save', help="every detection is saved here (for --replay; '' = off)")
    G('--bot-save', help='TurtleBot detection saved here (used by --replay)')
    G('--detect-log',
      help="every final detection (tray / TurtleBot) appended here as CSV: target "
           "xyz, camera xyz / height / distance, view angle, off-centre, marker px ('' = off)")
    G('--log', help="'' = off")
    G('--image-topic')
    G('--camera-calibration-file',
      help='camera intrinsics (null = mycobot_calibration/config/camera.yaml)')
    a.set_defaults(**cfg)
    argv, i = [], 0
    while i < len(sys.argv) - 1:          # '--relay-desk -0.2,0,90': a value starting with '-'
        t = sys.argv[1 + i]
        nxt = sys.argv[2 + i] if i + 2 < len(sys.argv) else None
        if t in ('--bot-place-offset', '--bot-box-offset', '--relay-spots', '--relay-desk',
                 '--relay-spots-bot',
                 '--relay-grid', '--relay-home') and nxt and nxt[:1] == '-':
            argv.append(f'{t}={nxt}')
            i += 2
        else:
            argv.append(t)
            i += 1
    if '--ros-args' in argv:              # ROS arguments (ros2 run / launch) are not ours
        argv = argv[:argv.index('--ros-args')]
    args, unknown = a.parse_known_args(argv)
    if unknown:                           # a typo must not run with the default silently
        import difflib
        opts = [o for act in a._actions for o in act.option_strings]
        hints = []
        for u in unknown:
            m = difflib.get_close_matches(u.split('=')[0], opts, n=1)
            hints.append(f'{u}' + (f' (did you mean {m[0]}?)' if m else ''))
        a.error('unknown option(s): ' + ', '.join(hints))
    args.tray = 0                   # the tray last measured (set by every detection)
    for o in ('relay_tray', 'bot_place_offset', 'bot_box_offset', 'bot_r_range', 'relay_grid',
              'relay_spot_r', 'relay_plan_pad', 'relay_grid_budget', 'relay_place_shift',
              'relay_home'):
        if isinstance(getattr(args, o), (int, float)):     # one number in the YAML
            setattr(args, o, [getattr(args, o)])
    sb = args.relay_spots_bot                        # YAML: a list of [x, y] or a string
    if isinstance(sb, str):
        sb = None if sb.strip().lower() == 'auto' else \
            [tuple(float(v) for v in q.split(',')) for q in sb.split(';') if q.strip()]
    args.relay_spots_bot = [tuple(map(float, q)) for q in sb] if sb else None
    args.regions = os.path.expanduser(args.regions)
    args.relay_plan_pads = sorted(set(args.relay_plan_pad), reverse=True)
    args.relay_plan_pad = args.relay_plan_pads[0]      # the one planned with right now
    for o, v in tray_types[args.tray_type].items():     # preset, unless given explicitly
        f = '--' + o.replace('_', '-')
        if o not in cfg_mine and not any(t == f or t.startswith(f + '=') for t in argv):
            setattr(args, o, list(v) if isinstance(v, (list, tuple)) else v)
    if args.camera_calibration_file is None:
        args.camera_calibration_file = os.path.join(
            get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml')
    args.holding = False            # App: starts empty (no put-back action here)
    if not args.spacing and args.gap:
        args.spacing = args.gap + args.tile_size
    if args.log:
        _tee_stdout(os.path.expanduser(args.log))
    # approach / grasp / first-pose heights are set by each detection
    args.approach, args.grasp_z, args.servo_height = 0.10, 0.04, 0.12
    args.refine, args.return_home = False, False
    show = not args.no_show and bool(os.environ.get('DISPLAY') or
                                     os.environ.get('WAYLAND_DISPLAY'))
    if not show:
        args.auto = True
        args.do = args.do or 'detect'

    rclpy.init()
    node = Cube(args.image_topic)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        app = HandleApp(node, args)
        app.say('ready' + ('' if args.go else ' (DRY RUN: add --go to move)'))
        first = {'detect': ('detect', app.act_detect),
                 'd11bot': ('D11 -> desk -> TurtleBot', app.act_d11_to_bot),
                 'botd11': ('TurtleBot -> desk -> D11', app.act_bot_to_d11),
                 'deskbot': ('desk -> TurtleBot', app.act_desk_to_bot),
                 'botdesk': ('TurtleBot -> desk', app.act_bot_to_desk)}
        if args.replay:
            app.submit('replay start', app.act_replay_start)
            app.jobs.join()
        if args.holding and not args.do:
            args.do = 'none'
        if args.search:
            app.submit('search pose', app.act_search, first.get(args.do, (None, None))[1])
        elif args.do not in ('', 'none'):
            app.submit(*first[args.do])
        if show:
            view(app)
        app.jobs.join()
    except KeyboardInterrupt:
        if 'app' in locals():
            app.stop()
    finally:
        if 'app' in locals() and args.go and app._settle_now not in (None, 'unavailable'):
            app.driver_settle('precise')      # other nodes expect the default
        cv2.destroyAllWindows()
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
