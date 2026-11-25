"""Command line: options -> HandleApp, window, --do first action."""
import argparse
import os
import threading

import cv2

from mycobot_tasks import regions as rg
import rclpy
from rclpy.executors import MultiThreadedExecutor

from mycobot_tasks.red_cube_hover import Cube
from mycobot_tasks.handle_pick.app import HandleApp
from mycobot_tasks.handle_pick.base import _tee_stdout
from mycobot_tasks.handle_pick.window import view


# tray presets (--tray-type): the values tray_handle_pick plans with
TRAY_TYPES = {
    # shelf tray (guide part E): body 150 x 100 x 56, handle bar 91 mm along the 10 cm side,
    # handle top 76 mm; 15 cm side along the box 20 cm side
    'shelf': dict(tray_height=0.076, relay_tray=(0.15, 0.10, 0.056), bot_yaw_offset=90.0,
                  grasp_depth=0.024, approach_above=0.036),
    # test tray: 16 x 10, handle top 72 mm, handle along the 16 cm side (fingers close along
    # the 10 cm side); 16 cm side along the box 20 cm side. Body height: not measured (56?)
    'test': dict(tray_height=0.072, relay_tray=(0.10, 0.16, 0.056), bot_yaw_offset=0.0,
                 grasp_depth=0.020, approach_above=0.04),
}


def main():
    from ament_index_python.packages import get_package_share_directory
    from mycobot_tasks import tray_handle_pick
    a = argparse.ArgumentParser(description=tray_handle_pick.__doc__.split('\n\n')[0])
    g = a.add_argument
    # markers / handle
    g('--marker-size', type=float, default=0.0225, help='black marker side [m]')
    g('--gap', type=float, default=0.024,
      help='handle length = gap between the inner edges of the two cut marker TILES '
           '(white margin included) [m]; centre spacing checked = gap + --tile-size '
           '(0 = no check)')
    g('--tile-size', type=float, default=0.030,
      help='cut marker tile incl. the white margin [m] (printed 30 mm)')
    g('--spacing', type=float, default=0.0,
      help='or the L-R marker centre spacing directly [m] (overrides --gap)')
    g('--max-spacing-err', type=float, default=0.004, help='refuse above this [m]')
    g('--max-tilt', type=float, default=10.0, help='marker plane tilt limit [deg]')
    g('--grasp-depth', type=float, default=0.024,
      help='finger tips this far below the handle top (marker plane) when grasping [m]')
    g('--approach-above', type=float, default=0.036,
      help='approach / lift: finger tips this far above the handle top [m]')
    g('--servo-above', type=float, default=0.08,
      help='first go with the finger tips this far above the handle top [m]')
    g('--min-approach-above', type=float, default=0.02,
      help='out of reach higher (tray far out, e.g. on the TurtleBot): first pose and '
           'approach / lift lowered down to this far above the handle top [m]')
    g('--grip-yaw-offset', type=float, default=90.0,
      help='TCP yaw relative to the L -> R line [deg]; fingers close along the TCP x '
           'axis, so 90 = across the bar')
    g('--table-z', type=float, default=None,
      help='z of the surface the tray stands on, base frame [m] (default: the shelf region z '
           'from regions.yaml, else the desk top -0.007; below the desk e.g. the floor -0.306 '
           'or a shelf fork)')
    g('--no-scene', dest='scene', action='store_false',
      help="don't load place_scene; use the old 1.5 x 1.5 m table box at --table-z instead")
    g('--scene-args', default='',
      help='place_scene options as one string, e.g. "--with-bot --bot 0.02,-0.40" or '
           '"--edge-points 0.05,-0.262,0.25,-0.245"')
    g('--max-handle-h', type=float, default=0.15,
      help='handle top higher than this over --table-z = wrong --table-z [m]')
    g('--table-margin', type=float, default=0.012, help='desk collision box raised by this [m]')
    g('--place-drop', type=float, default=0.0015,
      help='putting back: stop this far above the grasp height, then open [m]')
    g('--open-position', type=float, default=1.0, help='gripper open command (1.0 = 10 cm)')
    g('--grasp-open', type=float, default=0.5,
      help='opening around the handle: going down to grasp it and releasing it (0.5 = 5 cm; '
           'fully open fingers reach the TurtleBot beside its box). Detection still opens '
           '--open-position (fingers out of the camera view)')
    g('--hold-grip', type=float, default=0.2,
      help='gripper opening the legs with the tray held are planned with (collision check; '
           '0 = closed)')
    # detection
    g('--frames', type=int, default=5, help='camera frames per detection')
    g('--pin-timeout', type=float, default=4.0, help='wait for them [s]')
    g('--max-spread', type=float, default=0.004, help='midpoint xy jitter limit [m]')
    g('--no-track', dest='track', action='store_false',
      help="detection: don't move the camera (one marker / far / off-centre / at an angle)")
    g('--recenter', type=int, default=4, help='camera moves per detection at most')
    g('--min-marker-px', type=float, default=30.0,
      help='markers smaller than this [px side] = too far: look from closer')
    g('--max-off-centre', type=float, default=0.30,
      help='midpoint further from the image centre (fraction of the width) = recentre')
    g('--max-view', type=float, default=20.0,
      help='camera axis vs marker normal above this [deg] = look from straight above')
    g('--track-tol', type=float, default=0.01,
      help='estimate within this of the last look point = no further camera move [m]')
    g('--scan-step', type=float, default=0.08,
      help='nothing seen: look this far +-x / +-y around the search point [m]')
    g('--look-max-r', type=float, default=0.40,
      help='camera look point at most this far from the base [m]')
    g('--step-check', type=float, default=0.01,
      help='pick plan: vertical legs checked with chained IK every this [m]')
    g('--max-step-jump', type=float, default=12.0,
      help='... and refused if one step moves any joint more than this [deg] (branch flip)')
    g('--grip-settle', type=float, default=0.8,
      help='after closing wait this long, then read the finger gap [s]')
    g('--lift-split', action='store_true',
      help='pick + place: after grasping go up in the planned steps (approach, above, '
           'carry height) instead of trying one LIN to the carry height first')
    g('--max-j1-off', type=float, default=60.0,
      help='pick plan: J1 at most this far from the handle direction [deg]')
    g('--no-flip', action='store_true',
      help='pick plan: never use the reversed arm (J1 turned ~180 deg from the handle, the '
           'arm reaching over its back). By default it is the fallback when no normal '
           'solution is free, e.g. D11 beside the shelf rack: the upper arm (offset ~8.5 '
           'cm to one side) hits the rack in the normal pose, not in the reversed one')
    g('--min-grip-gap', type=float, default=1.5,
      help='closed gripper gap below this = nothing grasped [mm]')
    g('--full-open-mm', type=float, default=100.0,
      help='gripper gap at joint value 1.0 [mm]')
    # motion
    g('--planner', choices=['pilz', 'ompl'], default='pilz',
      help='pilz: straight lines (LIN) for every move near / with the tray, PTP otherwise; '
           'ompl: the old any-path planning')
    g('--lin-velocity', type=float, default=0.6,
      help='LIN moves: scaling of pilz_cartesian_limits (max_trans_vel 0.10 m/s)')
    g('--lin-accel', type=float, default=0.5,
      help='LIN moves: acceleration scaling (Pilz checks the joint limits along the line)')
    g('--velocity', type=float, default=0.6)
    g('--mid-settle', choices=['precise', 'once', 'fast'], default='precise',
      help='intermediate motions (lift, up, carry, back, home, over the handle / put spot): '
           'driver settle_mode. fast = no correction, done within the driver '
           'fast_tolerance_deg; once = at most 1 correction; precise (default) = as before, '
           'like the grasp / put down / camera views (settle + up to max_corrections)')
    g('--driver-node', default='mycobot_driver', help='driver node (settle_mode parameter)')
    g('--slow-velocity', type=float, default=0.25, help='near the tray: descent and lift')
    g('--grasp-velocity', type=float, default=None, help='descent only (default: slow)')
    g('--accel', type=float, default=0.5)
    g('--search-x', type=float, default=0.25)
    g('--search-y', type=float, default=0.0)
    g('--search-height', type=float, default=0.28)
    g('--search-min-height', type=float, default=0.18)
    g('--search', action='store_true', help='go to the search pose first (it becomes home)')
    g('--do', choices=['', 'none', 'detect', 'hover', 'pick', 'pickplace', 'find', 'bot',
                       'sweep', 'return', 'shelf', 'd11bot', 'botd11', 'deskbot', 'botdesk'],
      default='')
    g('--place-at', type=lambda t: t if t in ('bot', 'shelf') else
      [float(v) for v in t.split(',')],
      default=[0.0, -0.18],
      help="m (pick + place): put the handle midpoint down at this x,y in base [m] "
           "(default 0,-0.18: on the desk, 18 cm toward -y from the base); 'bot' = on the "
           "TurtleBot (its marker + --bot-place-offset); 'shelf' = the shelf region centre "
           "(its z = the surface)")
    g('--place-yaw', default=None,
      help="m / k: tray direction at the place = handle L -> R (ID 10 -> 11) angle in base "
           "[deg] (0 = along +x, 90 = along +y); 'bot' = TurtleBot marker +x + "
           "--bot-yaw-offset; default: keep the picked direction")
    g('--shelf-yaw', type=float, default=None,
      help="k (and m with --place-at shelf): tray direction when put down on the shelf = "
           "handle L -> R angle in base [deg] (0 = +x, 90 = +y), independent of "
           "--place-yaw (that one is for the TurtleBot); default: --place-yaw if it is a "
           "number, else keep the picked direction")
    g('--place-z', default='-0.007',
      help="m: z of the surface the tray is put on, base frame [m] (desk top -0.007); "
           "'auto' = TurtleBot marker z - --bot-marker-dz")
    # regions / TurtleBot
    g('--regions', default=rg.PATH,
      help='search regions (ros2 run mycobot_tasks regions, set in RViz): shelf = where d / '
           'f look when nothing is seen (the desk top for now), bot = where t / m look for the TurtleBot')
    g('--scan-height', type=float, default=None,
      help="region scan: camera this far over the region surface [m] (lower if unreachable); "
           "default: the region's camera height (regions.yaml, 0.28)")
    g('--scan-pull', type=float, default=0.24,
      help='region scan: a view out of reach straight down is taken obliquely from up to '
           'this much closer to the base [m] (0 = straight down only)')
    g('--no-last', dest='use_last', action='store_false',
      help="nothing seen: don't look at the last tray / TurtleBot position before the "
           'region scan')
    g('--chase', type=int, default=2,
      help='scan: a dark patch cut by the image border (a marker partly outside the image) '
           'is followed this many camera moves (0 = off)')
    g('--chase-dark', type=float, default=90.0,
      help='chase: pixels darker than this (0..255, and the Otsu level) count as marker black')
    g('--scan-overlap', type=float, default=0.35,
      help='region scan: neighbouring views overlap by this fraction of the footprint')
    g('--bot-id', type=int, default=0, help='TurtleBot ArUco ID (DICT_4X4_50)')
    g('--bot-marker-size', type=float, default=0.070, help='TurtleBot marker side [m]')
    g('--bot-marker-dz', type=float, default=0.0,
      help='TurtleBot marker plane this far ABOVE the surface the tray goes on [m] '
           '(0 = marker lies on that surface, e.g. on the box floor)')
    g('--bot-place-offset', type=lambda t: [float(v) for v in t.split(',')],
      default=[0.0, 0.0, 0.0],
      help='place spot (handle midpoint) from the marker centre, dx,dy[,dz] [m]: dx,dy along '
           'the marker x / y (--bot-offset-frame marker, ArUco x = to the right of the '
           'printed marker, y = up) or base x / y (base); dz raises (+) / lowers (-) the '
           'place height')
    g('--bot-offset-frame', choices=['marker', 'base'], default='marker',
      help='frame of --bot-place-offset dx,dy')
    g('--bot-home', action='store_true',
      help='after finding the TurtleBot go back to home before the tray search (default: '
           'search the tray right from there)')
    g('--bot-yaw-offset', type=float, default=90.0,
      help="--place-yaw bot: tray L -> R = marker +x turned by this [deg] (90: the shelf "
           "tray's 15 cm side along the box 20 cm side; 0 = the 16 x 10 test tray)")
    g('--bot-box-offset', type=lambda t: [float(v) for v in t.split(',')],
      default=[0.0, 0.0],
      help='scene: box centre from the marker centre, dx,dy in the marker frame [m]')
    g('--bot-keepout-h', type=float, default=0.222,
      help='while the TurtleBot pose is not known (finding it / the tray on it): its whole '
           'area is an obstacle this high over the floor (Burger 0.192 with the LiDAR on top '
           '+ 3 cm) and camera views keep the gripper tip over it [m]')
    g('--bot-keepout-margin', type=float, default=0.03,
      help='... the TurtleBot area (regions.yaml) grown by this on each side [m]')
    g('--no-bot-scene', dest='bot_scene', action='store_false',
      help="don't put the TurtleBot + box into the planning scene where the marker was found")
    g('--bot-reuse', action='store_true',
      help='m: find the TurtleBot once and reuse it (default: find it again on every m, '
           'starting at its last position)')
    g('--detect-log', default='~/.ros/detect_log.csv',
      help="every final detection (tray / TurtleBot / sweep) appended here as CSV: target "
           "xyz, camera xyz / height / distance, view angle, off-centre, marker px ('' = off)")
    g('--sweep-target', choices=['tray', 'bot'], default='tray',
      help='g (height sweep): measure the tray handle or the TurtleBot marker')
    g('--sweep-heights', type=lambda t: [float(v) for v in t.split(',')],
      default=[0.18, 0.22, 0.26, 0.30, 0.35, 0.40],
      help='g: camera heights over the target, straight down [m]')
    g('--sweep-offsets', type=lambda t: [float(v) for v in t.split(',')], default=[0.06],
      help='g: then camera this far beside the target (+-x, +-y; the target off the image '
           "centre) at the middle height [m] ('' style 0 = none)")
    g('--on-bot-r', type=float, default=0.10,
      help='m: a tray found this close to the TurtleBot place spot (at its height) is the one '
           'already on it: not picked [m]')
    g('--origin-save', default='~/.ros/tray_origin.yaml',
      help='m onto the TurtleBot saves where the tray was here; r puts it back there')
    g('--return-keep-yaw', action='store_true',
      help="r: keep the tray direction it has on the TurtleBot (default: the original one)")
    g('--return-again', action='store_true',
      help='r: even when that origin was already used once')
    g('--bot-save', default='~/.ros/turtlebot_last.yaml',
      help='TurtleBot detection saved here (used by --replay)')
    g('--tray-type', choices=list(TRAY_TYPES), default='shelf',
      help='tray preset (options given on the command line win): ' + '; '.join(
          f'{k}: ' + ' '.join(f'--{o.replace("_", "-")} {v}' for o, v in d.items())
          for k, d in TRAY_TYPES.items()))
    g('--tray-height', type=float, default=0.076,
      help='tray bottom -> handle top (marker plane) [m]: the place heights come from it')
    g('--tray-height-tol', type=float, default=0.015,
      help='warn when the measured handle top over --table-z differs more [m]')
    g('--carry-clear', type=float, default=0.03,
      help='m: carry the tray with its bottom this far over the place surface [m]')
    g('--bot-rim', type=float, default=0.004,
      help='TurtleBot box rim over the surface the tray goes on (marker = box floor) [m]: '
           'carrying in and scanning while holding keep the tray bottom --carry-clear over it')
    g('--place-retreat', type=float, default=0.05,
      help='after releasing the tray: open fingers this far straight toward the base before '
           'going home [m] (0 = off)')
    g('--carry-min-r', type=float, default=0.20,
      help='carry: a straight line passing closer than this to the base axis is replaced by '
           'an arc around the base [m] (0 = always straight)')
    g('--carry-arc-step', type=float, default=15.0, help='carry arc: a via point every [deg]')
    g('--carry-blend', type=float, default=0.02,
      help='carry high -> via* -> p_high: tried first as ONE blended Pilz LIN sequence (no '
           'stop at the via points), corners rounded within this [m] (also capped at 40%% of '
           'the shorter neighbouring segment); refused -> one LIN per leg as before (0 = '
           'always one LIN per leg)')
    g('--carry-step', type=float, default=0.02,
      help='m: carry line not plannable -> try this much higher [m] ...')
    g('--carry-tries', type=int, default=6, help='... this many heights')
    # relay: shelf D11 <-> TurtleBot through a desk spot (n / j, relay.py)
    g('--relay-tray', type=lambda t: [float(v) for v in t.split(',')],
      default=[0.15, 0.10, 0.056],
      help='n / j: held tray body for the collision checks: length along the finger '
           'closing direction, width along the handle bar, height [m] (shelf tray: guide '
           'part E 150 x 100 x 56)')
    g('--relay-tilt', type=float, default=10.0,
      help='n / j: carrying, the tray may tilt this much [deg] (grasp / put down level)')
    g('--relay-tilt-weight', type=float, default=0.05,
      help='n / j: pull toward a level tray inside --relay-tilt (0 = any tilt up to the limit)')
    g('--relay-tcp-speed', type=float, default=0.06,
      help='n / j: fastest finger tip / tray edge speed [m/s]')
    g('--relay-accel', type=float, default=0.15,
      help='n / j: speeding up / slowing down at most this [m/s2] (smooth start and stop of '
           'every path: what is in the tray must not slide)')
    g('--relay-lever', type=float, default=0.10,
      help='n / j: tool rotation counts as the motion of a point this far away (tray edge) [m]')
    g('--relay-pad', type=float, default=0.005,
      help='n / j: after planning, every path is checked densely again, once as planned '
           '(contact = refused) and once with the tray grown by this on every side: the '
           'spots with less clearance are listed [m]')
    g('--relay-plan-pad', type=lambda t: [float(v) for v in t.split(',')],
      default=[0.005, 0.003, 0.0],
      help='n / j / e / r: plan with the held tray grown by this on every side [m] (clearance '
           'from the arm base / shelf), several = tried in turn until one works (most first; '
           'default 0.005,0.003,0: 0 mm used = WARNING)')
    g('--relay-carry-z', type=float, default=0.125,
      help='n / j: finger tip height while carrying over the grid, base z [m]')
    g('--relay-pull-r', type=float, default=0.16,
      help='n / j: out of the shelf the tray is pulled toward the base to this radius [m]')
    g('--relay-spots', type=lambda t: None if t.strip().lower() == 'auto' else
      [tuple(float(v) for v in p.split(',')) for p in t.split(';') if p.strip()],
      default=None,
      help='n / j / e / r: desk spot(s) for the handle midpoint, "x,y;x,y;..." base [m] (the '
           'tray must lie fully on the desk), e.g. -0.22,0 behind the arm base; default '
           '(auto): a desk grid at --relay-spot-r. e looks for the tray only there (camera '
           'over each spot; auto: the first one), r puts it there')
    g('--relay-lift-max', type=float, default=0.15,
      help='n / j / e / r: empty-arm moves to a higher goal (home, start) first go straight up '
           'by up to this (tool orientation kept), then OMPL [m] (0 = off)')
    g('--relay-desk-clear', type=float, default=0.04,
      help='n / j / e / r: empty-arm moves (OMPL: to the start, arm switch, home) keep this far '
           'above the desk top [m] (0 = off)')
    g('--relay-desk-clear-r', type=float, default=0.10,
      help='--relay-desk-clear: a square this half size around the arm base is left out [m]')
    g('--relay-plan-timeout', type=float, default=120.0,
      help='n / j / e / r: one planning part longer than this = failed (ERROR, the arm back '
           'home) [s] (0 = no limit). The real robot plans ~3x slower than the simulation')
    g('--bot-close-view', type=float, default=0.28,
      help='r / j: the tray in the TurtleBot box measured again with the camera this high over '
           'it, once the Burger is in the scene (the search view is high: small markers, '
           'poor tilt) [m] (0 = off)')
    g('--bot-place-r', type=float, default=0.295,
      help='n / e (placing): the tray handle put no farther than this from the arm base (inside '
           'the --bot-r-range max, so r / j find it within range despite the marker error) [m]')
    g('--bot-place-in', type=float, default=0.015,
      help='n / e: the TurtleBot box centre up to this beyond --bot-place-r is still used, the '
           'tray moved toward the arm base inside the box to --bot-place-r [m] (box: >= 13 mm '
           'left to the walls at 15 mm, ~11 mm at 20 mm)')
    g('--relay-place-shift', type=lambda t: [float(v) for v in t.split(',')], default=[0.005, 0.01],
      help='n / e: no way down into the TurtleBot box at its centre -> the tray put this far '
           'off it, each value in turn (toward the arm base first) [m] (0 = centre only)')
    g('--bot-r-range', type=lambda t: [float(v) for v in t.split(',')], default=[0.23, 0.30],
      help='n / j / e / r: the TurtleBot box centre (marker / tray in it) must be this far from '
           'the arm base axis, min,max [m]; outside: ERROR at once, no planning (sim: 30.5 and '
           '31.1 cm fail, 22 cm fails)')
    g('--relay-grid-budget', type=lambda t: [int(v) for v in t.split(',')], default=[100],
      help='n / j / e / r: desk -> TurtleBot first tried quickly: turn at the desk spot only, '
           'each carry grid search stopped after this many points, one pass per value '
           '(e.g. 40,120); nothing -> everything without a cap (0 = no quick pass) [points]')
    g('--relay-end', choices=['home', 'stay'], default='home',
      help='n / j / e / r: after the hand-over back to the home pose (--relay-home), then '
           'the gripper closed; stay = the arm stays where it ended')
    g('--relay-home', type=lambda t: [float(v) for v in t.split(',')], default=[],
      help='n / j / e / r: the home pose they end in (and go to after an error before '
           'carrying), J1..J6 [deg]; empty = the arm pose when the program started (not '
           'moved by the camera re-centring of a detection)')
    g('--relay-spot-r', type=lambda t: [float(v) for v in t.split(',')], default=[0.20, 0.23],
      help='n / j: automatic desk spots: radius range from the base axis rmin,rmax [m]')
    g('--relay-max-spots', type=int, default=5,
      help='n / j: desk spots tried when desk -> TurtleBot does not work from one [count]')
    g('--relay-grid', type=lambda t: [float(v) for v in t.split(',')],
      default=[-0.30, 0.30, -0.30, 0.30],
      help='n / j: carry grid xmin,xmax,ymin,ymax [m] (2 cm steps)')
    g('--relay-min-r', type=float, default=0.12, help='n / j: grid points this close to '
      'the base axis are left out [m]')
    g('--relay-touch', type=float, default=0.02,
      help='n / j / e / r: tray bottom within this over the surface it is put on / taken from '
           '(D11 fork / desk top / TurtleBot box floor, not its walls): contacts with that '
           'surface are normal (not collisions) [m]; covers the shelf margin (1 cm) + the '
           'tray clearance (--relay-plan-pad)')
    g('--relay-speed', type=float, default=0.35,
      help='n / j / e / r: fastest joint speed of the relay paths, empty arm [rad/s]')
    g('--relay-home-speed', type=float, default=0.7,
      help='n / j / e / r: fastest joint speed of the empty way home after the hand-over '
           '(OMPL part; the straight-up part beside the tray stays at --relay-speed) [rad/s]')
    g('--relay-home-tcp-speed', type=float, default=0.12,
      help='n / j / e / r: the same, fastest finger tip speed (--relay-accel scaled alike) '
           '[m/s]')
    g('--relay-held-speed', type=float, default=0.35,
      help='n / j / e / r: the same while holding the tray (--relay-tcp-speed scaled alike) '
           '[rad/s] (0.25 tried on the real arm 11-25: no less error off the path)')
    g('--relay-grasp-tries', type=int, default=3,
      help='n / j: different grasp poses (IK branches) tried on D11 when the pull out from '
           'one hits the rack / the arm itself [count]')
    g('--sim-speed', type=float, default=1.0,
      help='SIMULATION only (fake driver): motions this many times faster (relay paths '
           'and MoveIt moves; e.g. 3 = 3x, 0.5 = half speed). The real robot ignores it')
    g('--relay-desk', type=lambda t: [float(v) for v in t.split(',')], default=None,
      help='e / r (desk <-> TurtleBot): x,y,yaw: the tray on the desk, handle midpoint [m] '
           'and L -> R [deg], base frame. r: where it is put (default: the planner picks a '
           'desk spot); e with --layout-poses (simulation, no camera): where it stands '
           '(default: a desk spot the planner picks); e on the real robot: measured')
    g('--relay-d11', default='~/.ros/layout_tray.yaml',
      help='n: where the camera looks for the tray on D11 first; j: where the tray goes on D11 '
           '(layout export: handle top xyz + L -> R yaw)')
    g('--layout-poses', action='store_true',
      help='n / j, SIMULATION: D11 tray / TurtleBot / tray in the box as arranged in RViz '
           '(ros2 run mycobot_tasks layout files, read at every n / j), no camera')
    g('--layout-dir', default='~/.ros',
      help='--layout-poses: folder of the layout_*.yaml files (layout exports)')
    g('--relay-urdf', default='',
      help="n / j: URDF file for the relay IK when robot_state_publisher does not answer")
    g('--save', default='~/.ros/tray_handle_last.yaml',
      help="every detection is saved here (for --replay; '' = off)")
    g('--replay', nargs='?', const='~/.ros/tray_handle_last.yaml', default=None,
      help='SIMULATION: use the saved detection instead of the camera and start from the '
           'arm pose of that detection (run with handeye.launch.py sim:=true)')
    g('--place-offset', type=lambda t: [float(v) for v in t.split(',')], default=None,
      help='m: or put it this far from the pick spot, dx,dy [m] (overrides --place-at)')
    g('--go', action='store_true', help='really move (else plan only)')
    g('--real', action='store_true', help='allow moving a real (non-fake) robot')
    g('--auto', action='store_true', help='no step questions')
    g('--holding', action='store_true', help='start as holding the tray (after a restart)')
    g('--no-show', action='store_true', help='no camera window')
    g('--image-topic', default='/camera/image_raw')
    g('--camera-calibration-file', default=os.path.join(
        get_package_share_directory('mycobot_calibration'), 'config', 'camera.yaml'))
    g('--log', default='~/.ros/tray_handle_pick.log', help="'' = off")
    import sys
    argv, i = [], 0
    while i < len(sys.argv) - 1:          # '--place-at -0.1,0.2': a value starting with '-'
        t = sys.argv[1 + i]
        nxt = sys.argv[2 + i] if i + 2 < len(sys.argv) else None
        if t in ('--place-offset', '--place-at', '--place-yaw', '--place-z', '--shelf-yaw',
                 '--bot-place-offset', '--bot-box-offset', '--relay-spots', '--relay-desk',
                 '--relay-grid') and nxt and nxt[:1] == '-':
            argv.append(f'{t}={nxt}')
            i += 2
        else:
            argv.append(t)
            i += 1
    args, _ = a.parse_known_args(argv)
    args.tray = 0                   # the tray last measured (set by every detection)
    args.relay_plan_pads = sorted(set(args.relay_plan_pad), reverse=True)
    args.relay_plan_pad = args.relay_plan_pads[0]      # the one planned with right now
    for o, v in TRAY_TYPES[args.tray_type].items():     # preset, unless given explicitly
        f = '--' + o.replace('_', '-')
        if not any(t == f or t.startswith(f + '=') for t in argv):
            setattr(args, o, list(v) if isinstance(v, (list, tuple)) else v)
    if not args.spacing and args.gap:
        args.spacing = args.gap + args.tile_size
    # 'auto' / 'bot' values are resolved from the TurtleBot marker before planning
    args.place_z_spec, args.place_yaw_spec = args.place_z, args.place_yaw
    args.place_z = -0.007 if args.place_z == 'auto' else float(args.place_z)
    args.place_yaw = (None if args.place_yaw in (None, 'bot') else float(args.place_yaw))
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
        first = {'detect': ('detect', app.act_detect), 'hover': ('above handle', app.act_hover),
                 'find': ('find tray', app.act_find_tray),
                 'bot': ('find TurtleBot', app.act_find_bot),
                 'sweep': ('height sweep', app.act_sweep),
                 'return': ('TurtleBot -> back where m picked it', app.act_return),
                 'shelf': ('TurtleBot -> shelf', app.act_bot_to_shelf),
                 'd11bot': ('D11 -> desk -> TurtleBot', app.act_d11_to_bot),
                 'botd11': ('TurtleBot -> desk -> D11', app.act_bot_to_d11),
                 'deskbot': ('desk -> TurtleBot', app.act_desk_to_bot),
                 'botdesk': ('TurtleBot -> desk', app.act_bot_to_desk),
                 'pick': ('pick', app.act_pick), 'pickplace': ('pick + place', app.act_pick_place)}
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
