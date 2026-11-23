# Layout campaign for the relay (n / j) around a saved layout (default verified_1120):
# robot (world turned), TurtleBot (distance / bearing / yaw) and shelf (shift / yaw / fork
# height) varied one at a time; layouts where the desk / TurtleBot / shelf overlap are skipped;
# both directions planned (dry run incl. the dense collision check) in the sim this shell
# talks to (own ROS_DOMAIN_ID); the ones that work: TurtleBot +-2 cm (n) = success rate.
# Files in OUT/<name>/ only (~/.ros untouched). Results: OUT/rank2.txt
#   ROS_DOMAIN_ID=87 python3 rank2.py OUT
import math, os, re, subprocess, sys, time
import numpy as np
from mycobot_tasks import layout as lay, place_scene as ps

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
BASE = lay.preset_path(os.environ.get('BASE', 'verified_1120'))
COMMON = ['--relay-plan-pad', '0.005', '--bot-yaw-offset', '90']
PERTURB = [(0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)]


# ---------------------------------------------------------------- layout changes
def bot_polar(r=None, az=None):
    def f(L):
        x, y, yaw = L.v['bot']
        r0, a0 = math.hypot(x, y), math.atan2(y, x)
        r1 = r if r is not None else r0
        a1 = math.radians(az) if az is not None else a0
        d = math.degrees(a1 - a0)               # keep the box facing the base the same way
        L.v['bot'] = [r1 * math.cos(a1), r1 * math.sin(a1), math.remainder(yaw + d, 360.0)]
    return f


def bot_shift(dx=0.0, dy=0.0):
    def f(L):
        x, y, yaw = L.v['bot']
        L.v['bot'] = [x + dx, y + dy, yaw]
    return f


def bot_yaw(to):
    def f(L):
        L.v['bot'][2] = to
    return f


def shelf_shift(dx=0.0, dy=0.0):
    def f(L):
        x, y = L.plate_centre()
        L.set_plate_centre(x + dx, y + dy, L.v['shelf'][2])
    return f


def shelf_yaw(to):
    def f(L):
        x, y = L.plate_centre()
        L.set_plate_centre(x, y, to)
    return f


def key(k, v):
    def f(L):
        L.v[k] = v
    return f


def turn(d):
    def f(L):
        L.turn_world(d)
    return f


VARIANTS = [
    ('base', []),
    ('bot_r22', [bot_polar(r=0.22)]), ('bot_r28', [bot_polar(r=0.28)]),
    ('bot_az0', [bot_polar(az=0.0)]), ('bot_az-25', [bot_polar(az=-25.0)]),
    ('bot_yaw180', [bot_yaw(180.0)]), ('bot_yaw240', [bot_yaw(240.0)]), ('bot_yaw300', [bot_yaw(300.0)]),
    ('shelf_x-3', [shelf_shift(dx=-0.03)]), ('shelf_x+3', [shelf_shift(dx=0.03)]),
    ('shelf_y+2', [shelf_shift(dy=0.02)]),
    ('shelf_yaw85', [shelf_yaw(85.0)]), ('shelf_yaw95', [shelf_yaw(95.0)]),
    ('fork_287', [key('fork_top', 0.287)]),
    ('robot_turn-90', [turn(-90.0)]), ('robot_turn+90', [turn(90.0)]),
    # limits (2026-11-23 afternoon)
    ('shelf_x-1', [shelf_shift(dx=-0.01)]), ('shelf_x+1', [shelf_shift(dx=0.01)]),
    ('shelf_x+2', [shelf_shift(dx=0.02)]), ('shelf_y-1', [shelf_shift(dy=-0.01)]),
    ('shelf_y+1', [shelf_shift(dy=0.01)]), ('shelf_yaw88', [shelf_yaw(88.0)]),
    ('shelf_yaw92', [shelf_yaw(92.0)]), ('fork_270', [key('fork_top', 0.270)]),
    ('bot_r23', [bot_polar(r=0.23)]), ('bot_r24', [bot_polar(r=0.24)]),
    ('bot_r30', [bot_polar(r=0.30)]), ('bot_yaw255', [bot_yaw(255.0)]),
    # below the modelled fork range (263 mm): would a lower D11 work?
    ('fork_255', [key('fork_top', 0.255)]), ('fork_245', [key('fork_top', 0.245)]),
    ('fork_235', [key('fork_top', 0.235)]), ('fork_260', [key('fork_top', 0.260)]),
]


# ---------------------------------------------------------------- overlap check
def boxes(objs, names):
    out = []
    for n in names:
        for c, s, y in objs.get(n, []):
            out.append((n, np.array(c, float), np.array(s, float), float(y)))
    return out


def overlap(b1, b2, flat=False):
    """yaw-only boxes: z ranges (unless flat) and the 2D separating axis test."""
    _, c1, s1, y1 = b1
    _, c2, s2, y2 = b2
    if not flat and (abs(c1[2] - c2[2]) >= (s1[2] + s2[2]) / 2):
        return False

    def corners(c, s, y):
        R = np.array([[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]])
        return [c[:2] + R @ np.array([u, v]) for u in (-s[0] / 2, s[0] / 2) for v in (-s[1] / 2, s[1] / 2)]
    k1, k2 = corners(c1, s1, y1), corners(c2, s2, y2)
    for y in (y1, y2):
        for ax in (np.array([math.cos(y), math.sin(y)]), np.array([-math.sin(y), math.cos(y)])):
            p1, p2 = [ax @ k for k in k1], [ax @ k for k in k2]
            if max(p1) <= min(p2) or max(p2) <= min(p1):
                return False
    return True


def clashes(L):
    objs, _ = ps.build(L.args())
    desk = boxes(objs, ('desk', 'desk_legs', 'robot_mount'))
    bot = boxes(objs, ('turtlebot', 'bot_box'))
    shelf = boxes(objs, ps.SHELF_NAMES)
    out = set()
    for b in bot:
        for d in desk:
            if overlap(b, d, flat=d[0] == 'desk'):   # not under the desk either
                out.add(f'{b[0]}-{d[0]}')
        for s in shelf:
            if overlap(b, s):
                out.add(f'{b[0]}-{s[0]}')
    for s in shelf:
        for d in desk:
            if overlap(s, d):
                out.add(f'{s[0]}-{d[0]}')
    return sorted(out)


# ---------------------------------------------------------------- runs
def make(name, mods):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    L = lay.Layout(BASE)
    for m in mods:
        m(L)
    L.path = os.path.join(d, 'layout.yaml')
    L.save()
    lay.FILES = {k: os.path.join(d, os.path.basename(p)) for k, p in lay.FILES.items()}
    L.export()
    return d, L


def run(d, L, do):
    t0 = time.time()
    try:
        r = subprocess.run(['ros2', 'run', 'mycobot_tasks', 'tray_handle_pick', '--layout-poses']
                           + COMMON + ['--layout-dir', d, f'--scene-args=--layout {L.path}',
                                       '--do', do, '--no-show', '--log', ''],
                           capture_output=True, text=True, timeout=600)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
        out += '\nTIMEOUT (600 s)'
    open(os.path.join(d, f'{do}.log'), 'w').write(out)
    ok = 'relay plan OK' in out and 'dense collision check OK' in out
    m = re.search(r'plan OK in (\d+) s: desk spot (\([^)]*\))', out)
    t = re.search(r'tray tilt <= ([\d.]+) deg', out)
    near = re.search(r'closer than \d+ mm in (\d+) paths', out)
    why = ''
    if not ok:
        f = re.findall(r'^\s+(?:desk spot|desk <->|D11 <->|regrasp|D11 (?:reversed|normal)).*$', out, re.M)
        why = (f[0].strip() if f else ('TIMEOUT' if 'TIMEOUT' in out else out.strip().splitlines()[-1]))[:140]
    return dict(ok=ok, wall=round(time.time() - t0), plan=int(m.group(1)) if m else None,
                spot=m.group(2) if m else None, tilt=float(t.group(1)) if t else None,
                near=int(near.group(1)) if near else 0, why=why)


log = open(os.path.join(OUT, 'rank2.txt'), 'a')


def say(s):
    print(s, flush=True)
    log.write(s + '\n')
    log.flush()


say(f'=== {time.strftime("%Y-%m-%d %H:%M")} base {BASE}, {" ".join(COMMON)}')
only = os.environ.get('ONLY')
res = []
for name, mods in VARIANTS:
    if only and name not in only.split(','):
        continue
    d, L = make(name, mods)
    bx, by, byaw = L.v['bot']
    fx, fy, fz = L.fork_point()
    desc = (f'bot ({bx:+.3f},{by:+.3f}) r {math.hypot(bx, by) * 100:.0f} yaw {byaw:.0f}; '
            f'D11 ({fx:+.3f},{fy:+.3f}) r {math.hypot(fx, fy) * 100:.0f}')
    c = clashes(L)
    if c:
        say(f'{name:14s} SKIP overlap {c[:3]}  [{desc}]')
        res.append(dict(name=name, skip=True))
        continue
    rn, rj = run(d, L, 'd11bot'), run(d, L, 'botd11')
    say(f"{name:14s} n {'OK  ' if rn['ok'] else 'FAIL'} j {'OK  ' if rj['ok'] else 'FAIL'}  plan {rn['plan']}/{rj['plan']} s "
        f"spot {rn['spot']} tilt {rn['tilt']} near {rn['near']}/{rj['near']}  [{desc}]"
        + (f"\n{'':16s}n: {rn['why']}" if not rn['ok'] else '') + (f"\n{'':16s}j: {rj['why']}" if not rj['ok'] else ''))
    pert = []
    if rn['ok'] and rj['ok'] and not os.environ.get('NOPERT'):
        for dx, dy in PERTURB:
            dp, Lp = make(f'{name}__bot{dx:+.2f}_{dy:+.2f}', mods + [bot_shift(dx, dy)])
            if clashes(Lp):
                say(f'{"":16s}bot {dx:+.2f},{dy:+.2f}: SKIP overlap')
                continue
            rp = run(dp, Lp, 'd11bot')
            pert.append(rp['ok'])
            say(f"{'':16s}bot {dx:+.2f},{dy:+.2f}: {'OK  ' if rp['ok'] else 'FAIL'} plan {rp['plan']} s {rp['why']}")
    res.append(dict(name=name, skip=False, n=rn, j=rj, pert=pert))

say('=== ranking: both directions, then TurtleBot +-2 cm success, fewer close paths, faster plan')
done = [r for r in res if not r['skip']]
order = sorted(done, key=lambda r: (-(r['n']['ok'] + r['j']['ok']), -sum(r['pert']),
                                    r['n']['near'] + r['j']['near'], r['n']['plan'] or 9999))
for i, r in enumerate(order, 1):
    rate = f"{sum(r['pert'])}/{len(r['pert'])}" if r['pert'] else '-'
    say(f"{i:2d}. {r['name']:14s} n {'OK' if r['n']['ok'] else '--'} j {'OK' if r['j']['ok'] else '--'}  "
        f"+-2 cm {rate:4s} near {r['n']['near']}/{r['j']['near']} plan {r['n']['plan']} s tilt {r['n']['tilt']}")
both = sum(1 for r in done if r['n']['ok'] and r['j']['ok'])
pt = [p for r in done for p in r['pert']]
say(f'layouts tested {len(done)} (skipped {len(res) - len(done)} overlapping): both directions OK '
    f'{both}/{len(done)}; TurtleBot +-2 cm on those: {sum(pt)}/{len(pt)}')
