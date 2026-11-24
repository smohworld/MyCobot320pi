# D11 tray pose tolerance for n (D11 -> desk -> TurtleBot): the scene (shelf / fork) stays as
# in the base layout, only the tray on D11 is moved (the real tray may stand off the layout
# pose): shelf X (along the rail), shelf Y (fork out), yaw. n is dry run (plan + dense check,
# both parts) in the sim this shell talks to (own ROS_DOMAIN_ID). j is not affected (it puts
# the tray at the layout pose). Files in OUT/<name>/ only (~/.ros untouched).
#   ROS_DOMAIN_ID=77 python3 d11tol.py OUT            (CASES=... e.g. "x+2,y-1/yaw+5")
import math, os, re, subprocess, sys, time
import yaml
from mycobot_tasks import layout as lay

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
BASE = lay.preset_path(os.environ.get('BASE', '~/.ros/layout.yaml'))
COMMON = os.environ.get('COMMON', '--bot-yaw-offset 90').split()


def parse(name):
    """'x+2,y-1,yaw+5' -> dX, dY [m] (shelf frame, cm in the name), dyaw [deg]"""
    d = dict(x=0.0, y=0.0, yaw=0.0)
    if name != 'base':
        for t in name.split(','):
            m = re.fullmatch(r'(x|y|yaw)([+-][\d.]+)', t)
            d[m.group(1)] = float(m.group(2))
    return d['x'] / 100, d['y'] / 100, d['yaw']


def make(name):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    L = lay.Layout(BASE)
    if os.environ.get('BOT'):                          # TurtleBot marker x,y [m], yaw [deg]
        L.v['bot'] = [float(v) for v in os.environ['BOT'].split(',')]
    L.path = os.path.join(d, 'layout.yaml')
    L.save()
    lay.FILES = {k: os.path.join(d, os.path.basename(p)) for k, p in lay.FILES.items()}
    L.export(float(os.environ.get('TH', '0.072')))     # tray bottom -> handle top [m]
    dX, dY, dyaw = parse(name)
    o = L.shelf_to_base(0.0, 0.0)
    ex = [a - b for a, b in zip(L.shelf_to_base(1.0, 0.0), o)]
    ey = [a - b for a, b in zip(L.shelf_to_base(0.0, 1.0), o)]
    f = os.path.join(d, 'layout_tray.yaml')
    t = yaml.safe_load(open(f))
    t['handle_xyz'][0] += dX * ex[0] + dY * ey[0]
    t['handle_xyz'][1] += dX * ex[1] + dY * ey[1]
    t['yaw_lr_deg'] = math.degrees(math.remainder(math.radians(t['yaw_lr_deg'] + dyaw), 2 * math.pi))
    yaml.safe_dump(t, open(f, 'w'))
    return d, L, t


def run(d, L):
    t0 = time.time()
    try:
        cmd = (['python3', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'perturb_run.py')]
               if os.environ.get('PERTURB_DEG') else ['ros2', 'run', 'mycobot_tasks', 'tray_handle_pick'])
        r = subprocess.run(cmd + ['--layout-poses']
                           + COMMON + ['--layout-dir', d, f'--scene-args=--layout {L.path}',
                                       '--relay-d11', os.path.join(d, 'layout_tray.yaml'),
                                       '--do', 'd11bot', '--no-show', '--log', ''],
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
        out += '\nTIMEOUT (900 s)'
    open(os.path.join(d, 'd11bot.log'), 'w').write(out)
    a = 'D11 -> desk: dense collision check OK' in out
    b = 'desk -> TurtleBot: dense collision check OK' in out
    sp = re.search(r'desk spot \(([^)]*)\)', out)
    tl = re.findall(r'tray tilt <= ([\d.]+) deg', out)
    why = ''
    if not (a and b):
        f = re.findall(r'^.*(?:failed|FAIL|TIMEOUT|Error|error).*$', out, re.M)
        why = (f[-1].strip() if f else out.strip().splitlines()[-1])[:160]
        g = re.findall(r'^\s+(?:desk spot|desk <->|D11 <->|regrasp|D11 (?:reversed|normal)|no |pull).*$', out, re.M)
        if g:
            why += ' | ' + g[0].strip()[:140]
    pert = []
    for m in re.finditer(r'^PERTURB ([\d.]+) deg (D11 -> desk|desk -> TurtleBot): (dense collision check OK|HIT:)'
                         r'((?:\n  .*)*)', out, re.M):
        leg = 'A' if m.group(2).startswith('D11') else 'B'
        pert.append(f"{m.group(1)}deg {leg} " + ('ok' if m.group(3).startswith('dense') else
                    'HIT ' + '; '.join(x.strip()[:70] for x in m.group(4).strip().splitlines()[:2])))
    return dict(pert=pert, a=a, b=b, wall=round(time.time() - t0), spot=sp.group(1) if sp else None,
                tilt=max(map(float, tl)) if tl else None, why=why)


log = open(os.path.join(OUT, 'd11tol.txt'), 'a')


def say(s):
    print(s, flush=True)
    log.write(s + '\n')
    log.flush()


DEFAULT = (['base'] + [f'x{s}{v}' for v in (1, 2, 3) for s in '+-']
           + [f'y{s}{v}' for v in (1, 2, 3) for s in '+-']
           + [f'yaw{s}{v}' for v in (3, 6, 10) for s in '+-'])
cases = os.environ['CASES'].split('/') if os.environ.get('CASES') else DEFAULT
say(f'=== {time.strftime("%Y-%m-%d %H:%M")} base {BASE}, {" ".join(COMMON)}')
for name in cases:
    d, L, t = make(name)
    h = t['handle_xyz']
    r = run(d, L)
    ok = 'OK  ' if r['a'] and r['b'] else ('FAIL' if not r['a'] else 'FAILb')
    say(f"{name:16s} n {ok} D11->desk {'ok' if r['a'] else '--'} desk->bot {'ok' if r['b'] else '--'} "
        f"{r['wall']:4d} s spot ({r['spot']}) tilt {r['tilt']}  [handle {h[0]*1000:+.0f},{h[1]*1000:+.0f} "
        f"yaw {t['yaw_lr_deg']:.0f}]" + (f"\n{'':18s}{r['why']}" if r['why'] else '')
        + ''.join(f"\n{'':18s}{x}" for x in r['pert']))
