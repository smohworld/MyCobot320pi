# Desk spot study for n (D11 -> desk -> TurtleBot) with the current code and defaults: the
# desk spot forced with --relay-spots (one per run), dry run (plan + dense check, both
# legs). Per spot: works?, clearance the planner kept (graded --relay-plan-pad), plan time,
# path time, poses closer than 5 mm. Analysis only: no package code is changed.
#   ROS_DOMAIN_ID=77 LAYOUT_DIR=<dir with layout*.yaml> python3 spotscan.py OUT
import math, os, re, subprocess, sys, time

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
D = os.path.abspath(os.environ['LAYOUT_DIR'])
SCENE = os.environ.get('SCENE', '--rack-trays')
SPOTS = [tuple(float(v) for v in p.split(',')) for p in os.environ['SPOTS'].split(';')] \
    if os.environ.get('SPOTS') else \
    [(x, y) for x in (-0.28, -0.24, -0.20, -0.16, -0.12, -0.08, -0.04)
     for y in (-0.24, -0.20, -0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08)
     if 0.14 <= math.hypot(x, y) <= 0.30]


def leg(out, name):
    """numbers of one leg ('D11 -> desk' / 'desk -> TurtleBot') from the log"""
    r = {}
    m = re.search(rf'^{re.escape(name)}: planned with (\d+) mm clearance', out, re.M)
    r['pad'] = int(m.group(1)) if m else (0 if re.search(
        rf'WARNING: {re.escape(name)} planned with NO clearance', out) else None)
    m = re.search(rf'relay plan {re.escape(name)} OK in (\d+) s', out)
    r['plan'] = int(m.group(1)) if m else None
    m = re.search(rf'^{re.escape(name)}: \d+ paths, \d+ points, about (\d+) s', out, re.M)
    r['run'] = int(m.group(1)) if m else None
    m = re.search(rf'^{re.escape(name)}: dense collision check OK[^\n]*\n((?:    .*\n)*)', out, re.M)
    r['ok'] = m is not None
    near = 0
    if m:
        for ln in m.group(1).splitlines():
            for k in re.findall(r'(\w+-\w+) x(\d+)', ln):
                if 'desk-relay_tray' != k[0] and 'relay_tray-desk' != k[0]:
                    near += int(k[1])
    r['near'] = near
    return r


log = open(os.path.join(OUT, 'spotscan.txt'), 'a')


def say(s):
    print(s, flush=True)
    log.write(s + '\n')
    log.flush()


say(f'=== {time.strftime("%Y-%m-%d %H:%M")} layout {D}, scene {SCENE}, {len(SPOTS)} spots')
rows = []
for x, y in SPOTS:
    t0 = time.time()
    name = f'{x:+.2f}_{y:+.2f}'
    try:
        p = subprocess.run(['ros2', 'run', 'mycobot_tasks', 'tray_handle_pick', '--layout-poses',
                            '--layout-dir', D, f'--scene-args=--layout {D}/layout.yaml {SCENE}',
                            '--relay-d11', f'{D}/layout_tray.yaml', f'--relay-spots={x},{y}',
                            '--do', 'd11bot', '--no-show', '--log', ''],
                           capture_output=True, text=True, timeout=900)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')) + '\nTIMEOUT'
    open(os.path.join(OUT, f'{name}.log'), 'w').write(out)
    A, B = leg(out, 'D11 -> desk'), leg(out, 'desk -> TurtleBot')
    ok = A['ok'] and B['ok']
    why = ''
    if not ok:
        f = re.findall(r'^\s+(?:D11 (?:reversed|normal)|regrasp|spot|desk spot)[^\n]*$', out, re.M)
        why = (f[0].strip() if f else out.strip().splitlines()[-1])[:110]
    sw = re.search(r'arm switch \(empty[^)]*\) \(\d+ pts, ([\d.]+) s', out)
    row = dict(x=x, y=y, r=math.hypot(x, y), ok=ok, A=A, B=B,
               sw=float(sw.group(1)) if sw else None, wall=round(time.time() - t0))
    rows.append(row)
    fmt = lambda v: '-' if v is None else str(v)
    say(f"({x:+.2f},{y:+.2f}) r {row['r'] * 100:4.1f}  {'OK  ' if ok else 'FAIL'}  "
        f"pad {fmt(A['pad'])}/{fmt(B['pad'])} mm  plan {fmt(A['plan'])}+{fmt(B['plan'])} s  "
        f"run {fmt(A['run'])}+{fmt(B['run'])} s  switch {fmt(row['sw'])} s  "
        f"near5 {A['near']}+{B['near']}" + (f"\n{'':20s}{why}" if why else ''))

say('=== working spots, best first: most clearance kept, fewest poses < 5 mm, shortest run + plan')
good = [r for r in rows if r['ok']]
good.sort(key=lambda r: (-min(r['A']['pad'] or 0, r['B']['pad'] or 0), r['A']['near'] + r['B']['near'],
                         (r['A']['run'] or 999) + (r['B']['run'] or 999) + (r['A']['plan'] or 999) + (r['B']['plan'] or 999)))
for i, r in enumerate(good, 1):
    say(f"{i:2d}. ({r['x']:+.2f},{r['y']:+.2f}) r {r['r'] * 100:.1f} cm  pad {r['A']['pad']}/{r['B']['pad']} mm  "
        f"near5 {r['A']['near']}+{r['B']['near']}  run {r['A']['run']}+{r['B']['run']} s  "
        f"plan {r['A']['plan']}+{r['B']['plan']} s  switch {r['sw']} s")
say(f'{len(good)}/{len(rows)} spots work')
