# Rank layouts for the relay (n / j): each layout = the current ~/.ros/layout.yaml with one
# change (files in OUT/<name>/, ~/.ros untouched), planned D11 -> TurtleBot (dry run, own sim),
# then the TurtleBot moved +-2 cm (x, y) for the ones that work (robustness), then j for the
# best. Results appended to OUT/rank.txt.
#   python3 rank.py OUT
import os, re, subprocess, sys, time
from mycobot_tasks import layout as lay

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
# base layout: BASE=<saved layout name or path> (default ~/final_project_ws/layouts/verified_1120.yaml);
# a fixed file, so editing the layout in RViz meanwhile changes nothing here
BASE = lay.preset_path(os.environ.get('BASE', 'verified_1120'))
COMMON = ['--relay-plan-pad', '0.005', '--bot-yaw-offset', '90']


def bot(dx=0.0, dy=0.0, dyaw=0.0, to=None):
    def f(L):
        b = L.v['bot']
        L.v['bot'] = [b[0] + dx, b[1] + dy, to if to is not None else b[2] + dyaw]
    return f


def key(k, val):
    def f(L):
        L.v[k] = val
    return f


def shelf(dx=0.0, dy=0.0):
    def f(L):
        s = L.v['shelf']
        L.v['shelf'] = [s[0] + dx, s[1] + dy, s[2]]
    return f


def turn(d):
    def f(L):
        L.turn_world(d)
    return f


VARIANTS = dict(current=lambda L: None, bot_yaw_270=bot(to=270.0),
                bot_x_p3=bot(dx=0.03), bot_x_m3=bot(dx=-0.03),
                bot_y_p3=bot(dy=0.03), bot_y_m3=bot(dy=-0.03),
                bot_yaw_p30=bot(dyaw=30), bot_yaw_m30=bot(dyaw=-30),
                fork_287=key('fork_top', 0.287), shelf_x_p2=shelf(dx=0.02),
                shelf_y_m2=shelf(dy=-0.02), face_shelf=turn(-90.0))
PERTURB = [(0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)]


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
                           capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
        out += '\nTIMEOUT'
    open(os.path.join(d, f'{do}.log'), 'w').write(out)
    ok = 'relay plan OK' in out and 'COLLISION' not in out
    m = re.search(r'plan OK in (\d+) s: desk spot (\([^)]*\))', out)
    t = re.search(r'tray tilt <= ([\d.]+) deg', out)
    near = re.search(r'closer than \d+ mm in (\d+) paths', out)
    fail = re.search(r'relay plan failed.*?\n(.*?)\n', out, re.S)
    return dict(ok=ok, wall=time.time() - t0, plan=int(m.group(1)) if m else None,
                spot=m.group(2) if m else None, tilt=float(t.group(1)) if t else None,
                near=int(near.group(1)) if near else 0,
                why=(fail.group(1).strip()[:160] if fail else ('TIMEOUT' if 'TIMEOUT' in out else '')))


rank = open(os.path.join(OUT, 'rank.txt'), 'a')


def say(s):
    print(s, flush=True)
    rank.write(s + '\n')
    rank.flush()


only = os.environ.get('ONLY')
res = {}
for name, mod in VARIANTS.items():
    if only and name not in only.split(','):
        continue
    d, L = make(name, [mod])
    r = run(d, L, 'd11bot')
    res[name] = dict(base=r, bot=[round(v, 3) for v in L.v['bot']], pert=[])
    say(f"{name:12s} n {'OK  ' if r['ok'] else 'FAIL'} plan {r['plan']} s, spot {r['spot']}, "
        f"tilt {r['tilt']}, near {r['near']} paths, bot {res[name]['bot']} {r['why']}")
    if not r['ok']:
        continue
    for dx, dy in PERTURB:
        dp, Lp = make(f'{name}__{dx:+.2f}_{dy:+.2f}', [mod, bot(dx=dx, dy=dy)])
        rp = run(dp, Lp, 'd11bot')
        res[name]['pert'].append(rp['ok'])
        say(f"   bot {dx:+.2f},{dy:+.2f}: {'OK  ' if rp['ok'] else 'FAIL'} plan {rp['plan']} s {rp['why']}")
say('=== ranking (n works, TurtleBot +-2 cm robustness, then fewer close paths, faster plan)')
order = sorted(res.items(), key=lambda kv: (not kv[1]['base']['ok'], -sum(kv[1]['pert']),
                                            kv[1]['base']['near'], kv[1]['base']['plan'] or 9999))
for name, v in order:
    b = v['base']
    say(f"{name:12s} {'OK  ' if b['ok'] else 'FAIL'} robust {sum(v['pert'])}/{len(PERTURB) if b['ok'] else 0} "
        f"plan {b['plan']} s tilt {b['tilt']} near {b['near']}")
for name, v in order[:3]:
    if not v['base']['ok']:
        continue
    d, L = make(name, [VARIANTS[name]])
    r = run(d, L, 'botd11')
    say(f"{name:12s} j {'OK  ' if r['ok'] else 'FAIL'} plan {r['plan']} s, near {r['near']} paths {r['why']}")
