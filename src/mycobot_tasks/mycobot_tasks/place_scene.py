"""MoveIt planning scene for picking / placing below the desk (TurtleBot3 Burger + box).

Layout (base frame, robot on a 29.9 cm desk, base 7 mm above the desk top):
  desk   60 x 110 cm: 60 cm along y from --edge-y (-0.31) to -0.31 + 0.60 = +0.29, i.e. the
         base is 31 cm from the -y (robot's right, drop-off) edge (user, 2026-11-18) and
         29 cm from the +y edge. The robot sits at the 60 cm end: the desk runs
         1.10 m back (-x) from just in front of the base (--desk-x -1.04,0.06).
         Desk frame: x along the 110 cm side, the drop-off edge on its -y side. The robot base
         is turned --robot-yaw deg (CCW, seen from above) from it; 0 = base x along the desk.
         Measure both with --edge-points: touch the edge at two points with the finger tip,
         read `ros2 run tf2_ros tf2_echo base gripper_tcp`, pass x1,y1,x2,y2 (base, m) ->
         --robot-yaw and --edge-y are computed from them.
  floor  --desk-height below the desk top (z = -0.306).
  legs   desk_legs (user, 2026-11-20): at both x ends a 2 cm thick wall over the whole desk
         width (real: two 2 x 28 cm plates, 2.8 cm from the -y edge and 2 mm from the +y edge,
         to be closed by walls for the TurtleBot's LiDAR), open under the desk in between;
         floor -> desk top (--leg-plate / --leg-gap / --leg-beam;
         the old solid 28 x 57 blocks: --leg-size 0.28,0.57).
  mount  robot_mount: two support plates beside the base (y = +-mount_y, 0.5 cm thick),
         from --mount-x (6 cm in front of the base axis, = the desk front end by default)
         3 cm out (+x), 3 cm over and 11 cm under the desk top. Fixed to the robot (base
         frame): they turn / move with it, not with the desk.
  bot    a low open box BEHIND a TurtleBot3 Burger (user, 2026-11-19): box inner floor
         --box-floor (0.026) over the floor, walls --box-depth (0.004) high, inner --box-inner,
         the ArUco marker (ID 0) at the box floor centre, box x = marker +x; a support block
         fills the 2-3 cm under it. The Burger (--bot-body L x W, --bot-height) stands on the
         --bot-side of the box (box frame, default +y: user 2026-11-19; the box 20 cm side
         along the Burger's back), --bot-gap away, its front pointing
         away from the box; its camera sticks out of the front: --bot-cam out,width,centre
         height (0.06, 0.07, 0.14), --bot-cam-h tall. Placed by the box centre (--bot
         cx,cy,yaw: from the marker in tray_handle_pick).
  shelf  SISYPHUS robot shelf on the floor (--with-shelf), dimensions from the 3D model of
         its build guide (artifact '로봇선반 제작 가이드', info 2026-11-19; part boxes
         measured from the model, checked 2026-11-20). Shelf frame: origin = base-plate FRONT-LEFT corner on the floor, +X to
         the right seen from the front, +Y back toward the rack, +Z up. Placed by --shelf
         x,y,yaw (base frame, yaw = shelf +X in base, deg CCW) or measured with
         --shelf-points x1,y1,x2,y2: finger tip on the front-left corner of the base plate,
         then on any point further right along its front edge (xy only).
         Moving parts: --shelf-x (X axis, mast centre from the plate centre, -0.097..+0.097),
         --fork-top (D11 fork plate top over the floor; a cell support top - 8 mm = 0.254 /
         0.419 / 0.584 for L1..L3; the model reaches 0.263..0.616 only, see FORK_TOP),
         --fork-y (fork out toward the rack, 0..0.16).
         Fork stack (D07 carriage, D08 deck + Y motor, ball rails, D24, load cell, D11
         100 x 140 x 13.8) hangs 113 mm below the fork top.
         The tray on the fork is NOT added unless --with-tray (it would block the grasp);
         its centre / handle are printed. tray_handle_pick then needs  --table-z <fork top
         z> (printed).
         Boxes are bounding boxes of the model parts, grown by --shelf-margin.

Replaces the 1.5 x 1.5 m 'calibration_table' (it blocks everything below the desk) with the
real desk. NOTE: click_move / look_toward / red_cube_hover / handeye add that big table back
when they start -> run this again afterwards.

  ros2 run mycobot_tasks place_scene                         # desk + floor (default)
  ros2 run mycobot_tasks place_scene --with-bot --bot 0.02,-0.40 --box-floor 0.197
  ros2 run mycobot_tasks place_scene --with-bot --bot 0.02,-0.40,-85    # cx,cy,yaw_deg
  ros2 run mycobot_tasks place_scene --robot-yaw 5 --desk-x -0.10,1.00
  ros2 run mycobot_tasks place_scene --edge-points 0.05,-0.262,0.25,-0.245 --print
  ros2 run mycobot_tasks place_scene --with-bot              # + TurtleBot and box
  ros2 run mycobot_tasks place_scene --with-shelf --shelf-points 0.12,-0.36,0.40,-0.37 \
      --shelf-x -0.097 --fork-top 0.254 --print              # shelf: check the numbers first
  ros2 run mycobot_tasks place_scene --with-shelf --shelf 0.12,-0.36,-2 --fork-top 0.254
  ros2 run mycobot_tasks place_scene --layout                # placement from ~/.ros/layout.yaml
  ros2 run mycobot_tasks place_scene --clear                 # remove everything added here
  ros2 run mycobot_tasks place_scene --print                 # numbers only, no ROS
"""
import argparse
import math
import sys
import threading


SHELF_NAMES = ('shelf_base', 'shelf_xaxis', 'shelf_mast', 'shelf_fork', 'shelf_rack',
               'shelf_tray')
NAMES = ('desk', 'floor', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box') + SHELF_NAMES

# Shelf geometry in the shelf frame [mm] (build guide; see the docstring for the frame).
LEVELS = (262.0, 427.0, 592.0)                  # cell support tops over the floor (L1..L3)
CELLS_X = ((106.0, 288.0), (300.0, 482.0))      # cell inner x; left L1 = hand-over cell
X_TRAVEL = 97.0                                 # X axis: mast centre 300 +- 97
CARRIAGE = (118.0, 471.0)                       # D07 bottom over the moving plate top (B4)
FORK_OVER_CARRIAGE = 79.0                       # D11 top - D07 bottom (guide 3D model)
FORK_TOP = tuple(66.0 + c + FORK_OVER_CARRIAGE for c in CARRIAGE)   # 263..616 over the floor
# shelf tray (guide part E): D13 frame 150 x 100 x 6, PP body 50, D31 handle arch in the
# middle (30 along X, 91 along Y) up to 76 over the tray bottom
TRAY = dict(len=150.0, wid=100.0, h=56.0, handle_x=30.0, handle_y=91.0, handle_z=(10.0, 76.0))



def quat_z(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def rot(th, x, y):
    c, s = math.cos(th), math.sin(th)
    return [c * x - s * y, s * x + c * y]


def from_edge_points(a):
    """Two base-frame points on the drop-off edge -> robot yaw [deg] and edge y (desk frame)."""
    x1, y1, x2, y2 = a.edge_points
    dx, dy = x2 - x1, y2 - y1
    if math.hypot(dx, dy) < 0.05:
        raise SystemExit('--edge-points: the two points must be >= 5 cm apart')
    th = -math.atan2(dy, dx)                  # desk x (along the edge) seen in base: angle -th
    if abs(th) > math.pi / 2:                 # pick the edge direction closest to base +x
        th -= math.copysign(math.pi, th)
    edge = rot(th, x1, y1)[1]
    if edge > 0:
        print('WARNING: these points are on the +y side of the base; the drop-off edge is '
              'expected on -y (check the points or the robot yaw)')
    return math.degrees(th), edge


def from_shelf_points(a):
    """Front-left corner + a point further right on the front edge -> [x, y, yaw_deg]."""
    x1, y1, x2, y2 = a.shelf_points
    if math.hypot(x2 - x1, y2 - y1) < 0.05:
        raise SystemExit('--shelf-points: the two points must be >= 5 cm apart')
    return [x1, y1, math.degrees(math.atan2(y2 - y1, x2 - x1))]


def shelf_boxes(a):
    """-> {name: [(x0, x1, y0, y1, z0, z1)]} in the shelf frame [mm], info (shelf frame)."""
    cx = 300.0 + a.shelf_x * 1000           # mast / moving plate centre
    F = a.fork_top * 1000                   # fork plate top over the floor
    ext = a.fork_y * 1000
    tb, pt = a.plate_t * 1000, a.plate_top * 1000
    tx, ty = cx + a.tray_dx * 1000, a.tray_y * 1000 + ext   # tray centre
    b = {
        'shelf_base': [
            (0, 600, 0, 450, 0, tb),                     # base plate 600 x 450
            (10, 150, 15, 135, tb, 38),                  # D20 electronics board + parts
            (517, 593, 20, 80, tb, 62),                  # D21 E-stop box
            (540, 570, 35, 65, 62, 80),                  # C1 E-stop button
            (175, 425, 112, 140, tb, 31),                # D15 X encoder bar
            (125, 145, 140, 154, tb, 28),                # D14 + X home switch
            (65, 115, 185, 265, tb, 54),                 # D01 X motor bracket
            (69, 111, 204, 246, 54, 102),                # X motor
            (482, 534, 205, 245, tb, 53),                # D02 idler holder
            (82, 519, 205, 245, 36, 54),                 # GT2 belt, pulley, idler
            (125, 475, 154, 196, tb, pt - 12),           # SK12 + Ø12 rods + SC12UU (front)
            (125, 475, 254, 296, tb, pt - 12),           # (back)
            (150, 180, 425, 475, tb, 22),                # D18 rack brackets
            (440, 470, 425, 475, tb, 22),
        ],
        'shelf_xaxis': [
            (cx - 70, cx + 70, 145, 305, pt - 12, pt),   # moving plate 140 x 160
            (cx - 40, cx + 67, 104, 145, 21, 68),        # D30AB / D30C + X sensors (front)
            (cx - 20, cx + 20, 206, 228, 36, pt - 12),   # D03 belt clamp
        ],
        'shelf_mast': [
            (cx - 75, cx + 75, 176, 188, pt, pt + 674),  # front panel 150 x 674
            (cx - 75, cx - 63, 188, 228, pt, pt + 674),  # ribs 40 deep
            (cx + 63, cx + 75, 188, 228, pt, pt + 674),
            (cx - 70, cx - 10, 145, 176, pt, pt + 40),   # D04 mast feet (front)
            (cx + 10, cx + 70, 145, 176, pt, pt + 40),
            (cx - 40, cx + 40, 188, 252, pt, pt + 54),   # D05 Z motor base + motor
            (cx - 61, cx + 61, 188, 221, 150, 650),      # SK8 + Ø8 rods (Z5, 500)
            (cx - 11, cx + 11, 201, 223, 122, 735),      # T8 lead screw
            (cx - 30, cx + 30, 188, 226, 705, 730),      # D06 608ZZ top housing
            (cx - 18, cx - 5, 188, 215, 150, 186),       # D22 Z bottom stopper
            (cx - 52, cx - 32, 188, 204, 190, 204),      # Z home switch
            (cx + 62, cx + 84, 228, 249, 200, 610),      # D15 Z encoder bar (right rib back)
            (cx + 75, cx + 95, 193, 223, 199, 516),      # D23 cable clips
            (cx + 28, cx + 52, 170, 176, 698, 722),      # MPU6050 (panel front)
        ],
        'shelf_fork': [                                  # moves with Z (F)
            (cx - 60, cx + 60, 199, 238, F - 79, F + 10),  # D07 Z carriage
            (cx - 60, cx + 60, 238, 257, F - 79, F - 71),  # D07 front flange (deck bolts)
            (cx + 54, cx + 92, 227, 287, F - 67, F + 6),   # D30Z + Z sensors
            (cx - 50, cx + 50, 227, 437, F - 75, F - 63),  # D08 Y deck plate (210 long)
            (cx - 50, cx - 24, 227, 437, F - 75, F - 32),  # D08 walls + outer ball rails
            (cx + 24, cx + 50, 227, 437, F - 75, F - 32),
            (cx - 38, cx + 26, 383, 437, F - 113, F - 75),  # D09 + Y motor under the deck
            # moves with the fork (--fork-y)
            (cx - 36, cx + 36, 243 + ext, 427 + ext, F - 76, F - 27),  # D24 + inner rails
            (cx - 7, cx + 7, 302 + ext, 382 + ext, F - 27, F - 14),    # A1 load cell
            (cx - 50, cx + 50, 287 + ext, 427 + ext, F - 14, F),       # D11 fork 100x140x13.8
        ],
        'shelf_rack': [
            (94, 494, 450, 610, 0, 12),                  # bottom
            (94, 494, 450, 610, 772, 784),               # top
            (482, 494, 450, 610, 12, 772),               # right side
            (288, 300, 450, 610, 12, 772),               # divider
            (94, 106, 450, 610, 330, 772),               # left panel (L1 left open below 330)
            (94, 494, 610, 614.5, 0, 784),               # back panel
            (94, 112, 450, 462, 12, 338),                # D25 IR beam posts (front, back)
            (94, 112, 590, 602, 12, 338),
            (94, 131, 450, 462, 12, 20),                 # post feet
            (94, 131, 590, 602, 12, 20),
        ],
    }
    D = 605.5                                            # D12 rails: rack front .. 155.5 deep
    for ci, (x0, x1) in enumerate(CELLS_X):              # pin side ledge 37, far side 26
        for li, lv in enumerate(LEVELS):
            if ci == 0 and li == 0:                      # L1 left = hand-over cell (D12hand)
                b['shelf_rack'] += [(x0, x0 + 37, 450, D, lv - 10, lv),
                                    (x0, x0 + 6, 450, D, lv - 30, 338)]
            else:
                b['shelf_rack'] += [(x0, x0 + 37, 450, D, lv - 10, lv),     # ledge
                                    (x0, x0 + 6, 450, D, lv - 30, lv + 8)]  # wall plate
            b['shelf_rack'] += [(x1 - 26, x1, 450, D, lv - 10, lv),
                                (x1 - 6, x1, 450, D, lv - 30, lv + 8),
                                (x1 - 26, x1 - 6, 579, 600, lv, lv + 20)]   # cell switch
    T = TRAY
    hx, hy, hz = T['handle_x'] / 2, T['handle_y'] / 2, T['handle_z']
    tray = [(tx - T['len'] / 2, tx + T['len'] / 2, ty - T['wid'] / 2, ty + T['wid'] / 2,
             F, F + T['h']),
            (tx - hx, tx + hx, ty - hy, ty + hy, F + hz[0], F + hz[1])]   # D31 handle arch
    if a.with_tray:
        b['shelf_tray'] = tray
    info = dict(F=F, tray=(tx, ty), handle_top=F + hz[1], cx=cx)
    return b, info


def place_shelf(a, floor_z):
    """Shelf boxes -> base frame primitives {name: [(centre, size, yaw)]}, info (base frame)."""
    ox, oy, yaw_deg = a.shelf
    yaw = math.radians(yaw_deg)
    m = a.shelf_margin * 1000
    boxes, si = shelf_boxes(a)

    def to_base(x, y):
        p = rot(yaw, x / 1000, y / 1000)
        return [ox + p[0], oy + p[1]]

    objs = {}
    for name, bl in boxes.items():
        objs[name] = [(to_base((x0 + x1) / 2, (y0 + y1) / 2) + [floor_z + (z0 + z1) / 2000],
                       [(x1 - x0 + 2 * m) / 1000, (y1 - y0 + 2 * m) / 1000,
                        (z1 - z0 + 2 * m) / 1000], yaw)
                      for x0, x1, y0, y1, z0, z1 in bl]
    info = dict(fork_top=floor_z + si['F'] / 1000, tray_xy=to_base(*si['tray']),
                handle_top=floor_z + si['handle_top'] / 1000,
                tray_yaw_deg=yaw_deg, shelf=si)
    return objs, info


def build(a):
    """-> {name: [(centre xyz, size xyz, yaw)]}, info dict."""
    th = math.radians(a.robot_yaw)            # desk frame -> base: rotate by -th
    desk_top = a.table_z
    floor_z = desk_top - a.desk_height
    m = a.margin
    x0, x1 = a.desk_x[0], a.desk_x[1] + a.margin          # margin past the front end too
    y0, y1 = a.edge_y - m, a.edge_y + a.desk_width + m    # margin past both long edges
    t_desk = 0.05
    objs = {
        # top raised by margin: the arm is only a few mm accurate
        'desk': [(rot(-th, (x0 + x1) / 2, (y0 + y1) / 2) + [desk_top + m - t_desk / 2],
                  [x1 - x0, y1 - y0, t_desk], -th)],
        'floor': [([0.0, 0.0, floor_z - t_desk / 2], [3.0, 3.0, t_desk], 0.0)],
    }
    if a.leg_size is not None and a.leg_size[0] > 0:
        # old model: one solid block at each x end, centred in y, floor -> desk top
        lx, ly = a.leg_size
        yc = a.edge_y + a.desk_width / 2
        objs['desk_legs'] = [
            (rot(-th, xc, yc) + [(floor_z + desk_top) / 2],
             [lx + 2 * m, ly + 2 * m, desk_top - floor_z], -th)
            for xc in (a.desk_x[0] + lx / 2, a.desk_x[1] - lx / 2)]
    elif a.leg_plate[0] > 0:
        # legs (user 2026-11-20): at each x end two plates --leg-plate (x thick, y wide), side
        # by side in y between --leg-gap (from the -y / +y desk edges), and a crossbar
        # --leg-beam (x, y) in the middle of the desk; floor -> desk top, grown by --leg-margin
        t, w = a.leg_plate
        lm = a.leg_margin
        y_lo = a.edge_y + a.leg_gap[0]
        y_hi = a.edge_y + a.desk_width - a.leg_gap[1]
        zc, h = (floor_z + desk_top) / 2, desk_top - floor_z
        legs = []
        span = y_hi - y_lo
        if w <= 0 or 2 * w >= span:               # one wall over the whole width
            plates = [((y_lo + y_hi) / 2, span)]
        else:                                     # two plates with a gap between them
            plates = [(y_lo + w / 2, w), (y_hi - w / 2, w)]
        for xc in (a.desk_x[0] + t / 2, a.desk_x[1] - t / 2):
            for yc, wy in plates:
                legs.append((rot(-th, xc, yc) + [zc], [t + 2 * lm, wy + 2 * lm, h], -th))
        if a.leg_beam[0] > 0:
            bx, by = a.leg_beam
            legs.append((rot(-th, sum(a.desk_x) / 2, (y_lo + y_hi) / 2) + [zc],
                         [bx + 2 * lm, by + 2 * lm, h], -th))
        objs['desk_legs'] = legs
    if a.mount_out > 0:
        # robot support plates beside the base (base frame: they move / turn with the
        # robot, not with the desk), from --mount-x --mount-out further in +x, --mount-up
        # over / --mount-down under the desk top, --mount-t thick along y, centres at
        # y = +-mount_y
        x_front = a.mount_x
        mm = a.mount_margin
        z0, z1 = desk_top - a.mount_down, desk_top + a.mount_up
        x0m, x1m = x_front - a.mount_in, x_front + a.mount_out
        objs['robot_mount'] = [
            ([(x0m + x1m) / 2, sy * a.mount_y, (z0 + z1) / 2],
             [x1m - x0m + 2 * mm, a.mount_t + 2 * mm, z1 - z0 + 2 * mm], 0.0)
            for sy in (1, -1)]
    info = {'desk_top': desk_top, 'floor': floor_z}
    if a.with_bot:
        bot = a.bot if a.bot else rot(-th, 0.0, a.edge_y - 0.15)   # straight out from the edge
        cx, cy = bot[0], bot[1]
        yaw = math.radians(bot[2]) if len(bot) > 2 else math.atan2(cy, cx)
        c, s = math.cos(yaw), math.sin(yaw)

        def at(u, v):          # box frame (u along the 20 cm side) -> base xy
            return [cx + c * u - s * v, cy + s * u + c * v]

        inner_floor = floor_z + a.box_floor
        bottom = inner_floor - a.wall
        rim = inner_floor + a.box_depth
        L, W, t = a.box_inner[0], a.box_inner[1], a.wall + a.wall_margin
        hz = (rim - bottom) / 2 + bottom
        hh = rim - bottom
        objs['bot_box'] = [
            # floor plate + the support under it, down to the room floor
            (at(0, 0) + [(floor_z + bottom + a.wall) / 2],
             [L + 2 * t, W + 2 * t, bottom + a.wall - floor_z], yaw),
            (at(+(L + t) / 2, 0) + [hz], [t, W + 2 * t, hh], yaw),                    # far wall
            (at(-(L + t) / 2, 0) + [hz], [t, W + 2 * t, hh], yaw),                    # near wall
            (at(0, +(W + t) / 2) + [hz], [L, t, hh], yaw),
            (at(0, -(W + t) / 2) + [hz], [L, t, hh], yaw),
        ]
        # the Burger on --bot-side of the box (box frame), front pointing away from it
        bl, bw = a.bot_body                  # along its heading, across
        m = a.bot_margin
        dx, dy = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}[a.bot_side]
        half_box = (L / 2 + t) if dx else (W / 2 + t)
        dc = half_box + a.bot_gap + bl / 2   # box centre -> Burger centre
        hyaw = yaw + math.atan2(dy, dx)      # Burger heading in base
        ox = at(dx * dc, dy * dc)
        cam_out, cam_w, cam_z = a.bot_cam
        cf = dc + bl / 2 + cam_out / 2       # camera block centre along the heading
        objs['turtlebot'] = [
            (ox + [floor_z + a.bot_height / 2], [bl + 2 * m, bw + 2 * m, a.bot_height], hyaw),
            (at(dx * cf, dy * cf) + [floor_z + cam_z],
             [cam_out + 2 * m, cam_w + 2 * m, a.bot_cam_h + 2 * m], hyaw)]
        near = at(-(L / 2 + t), 0)
        info.update(bot_xy=(cx, cy), yaw_deg=math.degrees(yaw), box_floor=inner_floor,
                    rim=rim, gap_to_edge=a.edge_y - rot(th, *near)[1])
    if a.with_shelf:
        if a.shelf is None:
            raise SystemExit('--with-shelf needs --shelf x,y,yaw or --shelf-points')
        sobjs, sinfo = place_shelf(a, floor_z)
        objs.update(sobjs)
        info['shelf'] = sinfo
    return objs, info


def _corners(c, s, y):
    cy, sy = math.cos(y), math.sin(y)
    return [(c[0] + cy * u - sy * v, c[1] + sy * u + cy * v)
            for u, v in ((s[0] / 2, s[1] / 2), (-s[0] / 2, s[1] / 2),
                         (-s[0] / 2, -s[1] / 2), (s[0] / 2, -s[1] / 2))]


def _overlap(p, q):
    """Two scene boxes (centre, size, yaw) intersect (z interval + separating axes in xy)."""
    (c1, s1, y1), (c2, s2, y2) = p, q
    if c1[2] + s1[2] / 2 <= c2[2] - s2[2] / 2 or c2[2] + s2[2] / 2 <= c1[2] - s1[2] / 2:
        return False
    A, B = _corners(c1, s1, y1), _corners(c2, s2, y2)
    for P in (A, B):
        for i in range(4):
            ex, ey = P[(i + 1) % 4][0] - P[i][0], P[(i + 1) % 4][1] - P[i][1]
            a = [-ey * x + ex * y for x, y in A]
            b = [-ey * x + ex * y for x, y in B]
            if max(a) <= min(b) or max(b) <= min(a):
                return False
    return True


def box_hits(objs, names, others):
    """-> set of the 'others' objects one of the 'names' objects intersects."""
    out = set()
    for n in names:
        for p in objs.get(n, []):
            for o in others:
                if o not in out and any(_overlap(p, q) for q in objs.get(o, [])):
                    out.add(o)
    return out


def report(objs, info, a):
    print(f"robot yaw {a.robot_yaw:+.1f} deg vs the desk, desk (desk frame) "
          f"x {a.desk_x[0]:+.2f}..{a.desk_x[1]:+.2f}, "
          f"y {a.edge_y:+.2f}..{a.edge_y + a.desk_width:+.2f}")
    print(f"desk top z {info['desk_top']:+.3f}  (collision top {info['desk_top'] + a.margin:+.3f}),"
          f" floor z {info['floor']:+.3f}")
    if 'box_floor' in info:
        print(f"bot box centre ({info['bot_xy'][0]:+.3f}, {info['bot_xy'][1]:+.3f}), "
              f"yaw {info['yaw_deg']:.0f} deg, inner floor z {info['box_floor']:+.3f}, "
              f"rim z {info['rim']:+.3f}")
        print(f"near box wall {info['gap_to_edge'] * 100:.1f} cm past the desk edge"
              + ('  <-- box under the desk edge!' if info['gap_to_edge'] < 0 else ''))
        d = math.hypot(*info['bot_xy'])
        print(f"box centre {d * 100:.0f} cm from the base axis; pin (near side) ~"
              f"{(d - a.box_inner[0] / 2 + 0.01) * 100:.0f} cm "
              f"(vertical-down reach below the desk: ~30-34 cm)")
    if 'shelf' in info:
        s = info['shelf']
        si = s['shelf']
        print(f"shelf corner ({a.shelf[0]:+.3f}, {a.shelf[1]:+.3f}), +X yaw {a.shelf[2]:+.1f} deg;"
              f" X {a.shelf_x * 1000:+.0f} mm (mast centre {si['cx']:.0f} mm from the left end),"
              f" fork top {a.fork_top * 1000:.0f} mm over the floor, fork out "
              f"{a.fork_y * 1000:.0f} mm")
        print(f"fork top z {s['fork_top']:+.3f}  ->  tray_handle_pick --table-z "
              f"{s['fork_top']:.3f}")
        tx, ty = s['tray_xy']
        d = math.hypot(tx, ty)
        print(f"tray (handle) centre ({tx:+.3f}, {ty:+.3f}), {d * 100:.0f} cm from the "
              f"base axis; tray +x yaw {s['tray_yaw_deg']:+.1f} deg, handle top z "
              f"{s['handle_top']:+.3f}"
              + ('  <-- beyond ~30 cm: top-down grasp unlikely' if d > 0.30 else ''))
        if abs(a.shelf_x) > X_TRAVEL / 1000 + 1e-9:
            print(f'WARNING: --shelf-x outside the X travel +-{X_TRAVEL:.0f} mm')
        if not FORK_TOP[0] <= a.fork_top * 1000 <= FORK_TOP[1]:
            print(f'WARNING: --fork-top outside ~{FORK_TOP[0]:.0f}..{FORK_TOP[1]:.0f} mm')
        if not 0 <= a.fork_y <= 0.16:
            print('WARNING: --fork-y outside 0..0.16')
        hits = box_hits(objs, [n for n in objs if n.startswith('shelf')],
                        ['desk', 'desk_legs', 'robot_mount', 'turtlebot', 'bot_box'])
        if hits:
            print('WARNING: the shelf runs into ' + ', '.join(sorted(hits))
                  + ' -> move it (--shelf / layout)')
        if s['fork_top'] + a.table_margin_tray > a.table_z - 0.02:
            print('WARNING: tray_handle_pick adds a 1.5 x 1.5 m "calibration_table" at '
                  '--table-z; with '
                  'the fork top this high it reaches the robot base -> lower the fork '
                  '(--fork-top) or shrink that box in red_cube_hover.set_table')
    for n, prims in objs.items():
        for c, s, y in prims:
            print(f'  {n:9s} centre ({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) '
                  f'size ({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}) yaw {math.degrees(y):.0f}')


# RViz colours (MotionPlanning > Scene Geometry), r, g, b, a
COLORS = {
    'desk': (0.62, 0.45, 0.28, 1.0),          # wood brown
    'desk_legs': (0.42, 0.29, 0.17, 1.0),     # darker brown
    'floor': (0.55, 0.55, 0.55, 0.35),        # light grey, see-through
    'robot_mount': (0.25, 0.25, 0.25, 1.0),   # dark grey
    'turtlebot': (0.15, 0.35, 0.95, 1.0),     # blue
    'bot_box': (0.55, 0.75, 1.0, 1.0),        # light blue
    'shelf_base': (0.25, 0.70, 0.35, 1.0),    # shelf: greens
    'shelf_xaxis': (0.10, 0.50, 0.20, 1.0),
    'shelf_mast': (0.40, 0.80, 0.45, 1.0),
    'shelf_rack': (0.20, 0.60, 0.30, 0.85),
    'shelf_fork': (1.0, 0.55, 0.10, 1.0),     # Z carriage / Y deck / D11: orange
    'shelf_tray': (1.0, 0.90, 0.15, 1.0),     # tray on D11: yellow
}


def object_colors(names):
    """moveit_msgs ObjectColor list for these object ids (the ones in COLORS)."""
    from moveit_msgs.msg import ObjectColor
    from std_msgs.msg import ColorRGBA
    return [ObjectColor(id=n, color=ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3]))
            for n in names for c in [COLORS.get(n)] if c is not None]


def collision_objects(objs, base='base'):
    """{name: [(centre, size, yaw)]} -> moveit_msgs CollisionObject ADD list."""
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
    cos = []
    for n, prims in objs.items():
        co = CollisionObject()
        co.header.frame_id = base
        co.id = n
        co.operation = CollisionObject.ADD
        for c, s, y in prims:
            co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX,
                                                dimensions=[float(v) for v in s]))
            p = Pose()
            p.position.x, p.position.y, p.position.z = (float(v) for v in c)
            (p.orientation.x, p.orientation.y, p.orientation.z,
             p.orientation.w) = quat_z(y)
            co.primitive_poses.append(p)
        cos.append(co)
    return cos


def apply(objs, clear, base='base', keep_table=False, wait=10.0):
    import rclpy
    from moveit_msgs.msg import CollisionObject, PlanningScene
    from moveit_msgs.srv import ApplyPlanningScene
    from rclpy.executors import MultiThreadedExecutor

    from mycobot_calibration.moveit_client import _wait

    rclpy.init()
    node = rclpy.create_node('place_scene')
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        cli = node.create_client(ApplyPlanningScene, 'apply_planning_scene')
        if not cli.wait_for_service(timeout_sec=wait):
            raise SystemExit('apply_planning_scene not available (is move_group running?)')
        cos = []
        for n in (() if keep_table else ('calibration_table',)) + NAMES:   # start clean
            co = CollisionObject()
            co.header.frame_id = base
            co.id = n
            co.operation = CollisionObject.REMOVE
            cos.append(co)
        if not clear:
            cos += collision_objects(objs, base)
        # removes one at a time (a diff removing an object that does not exist fails as a
        # whole, so that failure is ignored), then all adds in one diff
        def send(batch):
            req = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True))
            req.scene.world.collision_objects = batch
            req.scene.object_colors = object_colors(
                [co.id for co in batch if co.operation == CollisionObject.ADD])
            res = _wait(cli.call_async(req), 5.0)
            return res is not None and res.success
        nrm = len(NAMES) + (0 if keep_table else 1)
        for co in cos[:nrm]:
            send([co])
        if cos[nrm:] and not send(cos[nrm:]):
            raise SystemExit('apply_planning_scene failed')
        print('scene cleared' if clear else f'scene applied: {", ".join(objs)} '
              + ('(calibration_table kept)' if keep_table else '(calibration_table removed)'))
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


LIST_OPTS = ('--shelf', '--shelf-points', '--edge-points', '--bot', '--desk-x',
             '--box-inner', '--bot-body', '--leg-size', '--bot-cam', '--leg-plate',
             '--leg-gap', '--leg-beam')


def join_negative(argv):
    """'--shelf -0.3,0.1,0' -> '--shelf=-0.3,0.1,0' (argparse takes '-0.3,...' for an option)."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        if (t in LIST_OPTS and i + 1 < len(argv) and argv[i + 1][:1] == '-'
                and argv[i + 1][1:2] in '0123456789.'):
            out.append(f'{t}={argv[i + 1]}')
            i += 2
        else:
            out.append(t)
            i += 1
    return out


def floats(s):
    return [float(v) for v in s.split(',')]


def make_parser():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--table-z', type=float, default=-0.007, help='desk top z in base [m]')
    a.add_argument('--margin', type=float, default=0.012,
                   help='desk collision box raised / pushed out past the edge by this [m]')
    a.add_argument('--desk-height', type=float, default=0.299, help='desk top over the floor')
    a.add_argument('--edge-y', type=float, default=-0.36,
                   help='desk -y (robot right, drop-off) edge [m]: base 31 cm from it '
                        '(2026-11-18)')
    a.add_argument('--desk-width', type=float, default=0.60, help='desk size along y [m]')
    a.add_argument('--desk-x', type=floats, default=[-1.04, 0.06],
                   help='desk x range xmin,xmax in the desk frame [m]. Default: the robot sits '
                        'at the 60 cm end, the desk runs 1.10 m back (-x), its end edge 6 cm in '
                        'front of the base axis (measured 2026-11-16)')
    a.add_argument('--robot-yaw', type=float, default=0.0,
                   help='base +x vs the desk long side, CCW seen from above [deg]')
    a.add_argument('--edge-points', type=floats, default=None,
                   help='x1,y1,x2,y2: two finger-tip points on the drop-off edge (base, m) '
                        '-> sets --robot-yaw and --edge-y')
    a.add_argument('--leg-size', type=floats, default=None,
                   help='old leg model: one solid block x,y [m] at each x end (e.g. 0.28,0.57); '
                        'given = used instead of the plates below')
    a.add_argument('--leg-plate', type=floats, default=[0.02, 0.0],
                   help='desk legs: at each x end x thick, y wide [m]; y 0 = one wall over the '
                        'whole width (the real 2 x 28 cm plates + gaps, closed by walls for the '
                        "TurtleBot's LiDAR; two plates: e.g. 0.02,0.28 --leg-gap 0.028,0.002); "
                        '0,0 = no legs')
    a.add_argument('--leg-gap', type=floats, default=[0.0, 0.0],
                   help='legs: gap from the -y (right) and +y (left) desk edges [m] (real: '
                        '0.028, 0.002)')
    a.add_argument('--leg-beam', type=floats, default=[0.0, 0.0],
                   help='legs: crossbar in the middle of the desk, x,y [m] (0,0 = none: the '
                        'desk is open between its ends, user 2026-11-20)')
    a.add_argument('--leg-margin', type=float, default=0.005, help='legs grown by this [m]')
    a.add_argument('--mount-y', type=float, default=0.074,
                   help='robot support plates: centre |y| from the base axis [m] (user 2026-11-20: 14.8 cm apart; the base '
                        'is 11.7 cm wide: plates closer than ~6.2 cm touch it in the scene)')
    a.add_argument('--mount-x', type=float, default=0.06,
                   help='robot support plates start this far in front of the base axis (+x, '
                        'base frame) [m]: fixed to the robot (default = the desk front end)')
    a.add_argument('--mount-t', type=float, default=0.005, help='support plate thickness (y) [m]')
    a.add_argument('--mount-out', type=float, default=0.03,
                   help='support sticks out this far past --mount-x (+x) [m] '
                        '(0 = no supports)')
    a.add_argument('--mount-in', type=float, default=0.0,
                   help='... and reaches this far back over the desk (-x) [m]')
    a.add_argument('--mount-up', type=float, default=0.03, help='support top over the desk top [m]')
    a.add_argument('--mount-down', type=float, default=0.11,
                   help='support bottom under the desk top [m] (14 cm tall in all)')
    a.add_argument('--mount-margin', type=float, default=0.003,
                   help='support plates grown by this [m] (small: they sit next to the base)')
    a.add_argument('--bot', type=floats, default=None,
                   help='box centre in base cx,cy[,yaw_deg]; yaw = box 20 cm side (marker +x). '
                        'Default: 15 cm out from the edge, straight out from the base, '
                        'pointing away from it')
    a.add_argument('--box-inner', type=floats, default=[0.20, 0.16], help='inner L,W [m]')
    a.add_argument('--box-depth', type=float, default=0.004, help='inner depth (walls) [m]')
    a.add_argument('--box-floor', type=float, default=0.026,
                   help='box inner floor (marker plane) height over the room floor [m] '
                        '(box behind the Burger, 26 mm: user 2026-11-19)')
    a.add_argument('--wall', type=float, default=0.003, help='box wall / bottom thickness [m]')
    a.add_argument('--wall-margin', type=float, default=0.004,
                   help='box walls made this much thicker (outward) in the scene [m]')
    a.add_argument('--bot-body', type=floats, default=[0.138, 0.178],
                   help='Burger body L (along its heading), W [m] (TurtleBot3 Burger spec)')
    a.add_argument('--bot-height', type=float, default=0.192, help='Burger height [m]')
    a.add_argument('--bot-side', choices=['+x', '-x', '+y', '-y'], default='+y',
                   help='Burger on this side of the box (box frame = marker frame); its front '
                        'points that way too')
    a.add_argument('--bot-gap', type=float, default=0.0,
                   help='box outer wall -> Burger back [m]')
    a.add_argument('--bot-cam', type=floats, default=[0.06, 0.07, 0.14],
                   help='Burger front camera: out past the front, width, centre height over '
                        'the floor [m]')
    a.add_argument('--bot-cam-h', type=float, default=0.04,
                   help='camera block height [m] (not measured: a guess)')
    a.add_argument('--bot-margin', type=float, default=0.01,
                   help='Burger body / camera grown by this in the scene [m]')
    a.add_argument('--with-bot', action='store_true',
                   help='also add the TurtleBot + box (default: desk and floor only)')
    a.add_argument('--keep-table', action='store_true',
                   help="keep 'calibration_table' (e.g. tray_handle_pick --table-z <lower "
                        "surface> adds "
                        'it at the surface the tray stands on)')
    s = a.add_argument_group('shelf')
    s.add_argument('--with-shelf', action='store_true', help='add the robot shelf')
    s.add_argument('--shelf', type=floats, default=None,
                   help='x,y,yaw_deg: base-plate front-left corner (base, m) and the shelf +X '
                        '(to the right seen from the front) in base, deg CCW')
    s.add_argument('--shelf-points', type=floats, default=None,
                   help='x1,y1,x2,y2: finger tip on the front-left corner, then further right '
                        'on the front edge (base, m) -> sets --shelf')
    s.add_argument('--shelf-x', type=float, default=0.0,
                   help='X axis: mast centre from the plate centre [m], -0.097..+0.097 '
                        '(negative = to the left)')
    s.add_argument('--fork-top', type=float, default=0.254,
                   help='fork plate top over the floor [m] (= tray bottom; L1/L2/L3 hand-over '
                        'height 0.254/0.419/0.584)')
    s.add_argument('--fork-y', type=float, default=0.0,
                   help='fork moved out toward the rack [m], 0..0.16')
    s.add_argument('--tray-y', type=float, default=0.357,
                   help='tray centre (shelf Y from the front edge) with the fork in [m] '
                        '(default = D11 centre in the guide model; measure it)')
    s.add_argument('--tray-dx', type=float, default=0.0,
                   help='tray centre - mast centre along the shelf X [m] (measure it)')
    s.add_argument('--with-tray', action='store_true',
                   help='also add the tray on the fork as a collision object (not for picking)')
    s.add_argument('--plate-t', type=float, default=0.012, help='base plate thickness [m]')
    s.add_argument('--plate-top', type=float, default=0.066,
                   help='moving plate top over the floor [m] (Z 184 mm - carriage 118 mm)')
    s.add_argument('--shelf-margin', type=float, default=0.01,
                   help='shelf boxes grown by this on every side [m]')
    s.add_argument('--table-margin-tray', type=float, default=0.012,
                   help="tray_handle_pick's --table-margin (only for the warning)")
    a.add_argument('--layout', nargs='?', const=LAYOUT, default=None,
                   help='take desk / TurtleBot / shelf placement and the shelf axes from '
                        f'this layout file (default {LAYOUT}, set by `ros2 run mycobot_tasks '
                        'layout` in RViz); options given here override it')
    a.add_argument('--clear', action='store_true', help='remove the objects of this scene')
    a.add_argument('--print', action='store_true', help='print the numbers only, no ROS')
    a.add_argument('--base-frame', default='base')
    a.add_argument('--wait', type=float, default=10.0,
                   help='wait this long for move_group (apply_planning_scene) [s]')
    return a


LAYOUT = '~/.ros/layout.yaml'
# keys of the layout file (ros2 run mycobot_tasks layout) = place_scene option dests
LAYOUT_KEYS = ('desk_x', 'edge_y', 'robot_yaw', 'bot', 'with_bot', 'shelf', 'with_shelf',
               'shelf_x', 'fork_top', 'fork_y')


def load_layout(path=LAYOUT):
    """The layout file -> {option dest: value} (only LAYOUT_KEYS; {} when missing)."""
    import os
    import yaml
    try:
        with open(os.path.expanduser(path)) as f:
            d = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {k: d[k] for k in LAYOUT_KEYS if k in d and d[k] is not None}


def parse(argv):
    """place_scene options (list of strings) -> args with --edge-points / --shelf-points
    resolved. --layout: the layout file gives the defaults (options given here win)."""
    p = make_parser()
    argv = join_negative(list(argv))
    pre, _ = p.parse_known_args(argv)
    if pre.layout:
        d = load_layout(pre.layout)
        if not d:
            print(f'WARNING: --layout {pre.layout}: no layout there (ros2 run mycobot_tasks '
                  'layout), defaults used')
        p.set_defaults(**d)
    args, _ = p.parse_known_args(argv)

    if args.edge_points:
        args.robot_yaw, args.edge_y = from_edge_points(args)
        print(f'from --edge-points: --robot-yaw {args.robot_yaw:.2f} --edge-y {args.edge_y:.3f}')
    if args.shelf_points:
        args.shelf = from_shelf_points(args)
        print('from --shelf-points: --shelf ' + ','.join(f'{v:.3f}' for v in args.shelf))
    if args.shelf is not None and len(args.shelf) == 2:
        args.shelf = list(args.shelf) + [0.0]
    return args


def main():
    args = parse(sys.argv[1:])
    objs, info = build(args)
    report(objs, info, args)
    if not args.print:
        apply(objs, args.clear, args.base_frame, args.keep_table, args.wait)


if __name__ == '__main__':
    main()
