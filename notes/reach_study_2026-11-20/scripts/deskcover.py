# e coverage: which tray poses on the desk (handle midpoint x, y, L->R yaw) can e
# (desk -> TurtleBot) plan from, with the current code and defaults. Dry run per pose
# (--layout-poses --relay-desk x,y,yaw --do deskbot, no --go): plan only, nothing moves.
# The Burger from the layout files (layout_bot.yaml). Analysis only.
#   ROS_DOMAIN_ID=77 HOME=<sim home> python3 deskcover.py OUT.csv [--step 0.04] [--yaws 0,45,90,135]
import argparse, csv, math, os, re, subprocess, time

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--step', type=float, default=0.04)
ap.add_argument('--yaws', default='0,45,90,135')
ap.add_argument('--r', default='0.13,0.31', help='handle radius range from the base [m]')
ap.add_argument('--x', default='-0.30,0.02')
ap.add_argument('--y', default='-0.32,0.10')
ap.add_argument('--timeout', type=float, default=120)
ap.add_argument('--extra', default='', help='more tray_handle_pick options')
ap.add_argument('--do', default='deskbot', help='deskbot (e: pick from the desk) or botdesk (r: put down there)')
a = ap.parse_args()

r0, r1 = map(float, a.r.split(','))
x0, x1 = map(float, a.x.split(','))
y0, y1 = map(float, a.y.split(','))
n = lambda lo, hi: int(round((hi - lo) / a.step)) + 1
pts = [(round(x0 + i * a.step, 3), round(y0 + j * a.step, 3))
       for i in range(n(x0, x1)) for j in range(n(y0, y1))]
pts = [p for p in pts if r0 <= math.hypot(*p) <= r1]
yaws = [float(v) for v in a.yaws.split(',')]
done = set()
if os.path.exists(a.out):
    with open(a.out) as f:
        done = {(float(r['x']), float(r['y']), float(r['yaw'])) for r in csv.DictReader(f)}
new = not os.path.exists(a.out)
f = open(a.out, 'a', newline='')
w = csv.writer(f)
if new:
    w.writerow(['x', 'y', 'yaw', 'ok', 'plan_s', 'pad_mm', 'why', 'far'])
print(f'{len(pts)} positions x {len(yaws)} yaws', flush=True)
for (x, y) in pts:
    for yaw in yaws:
        if (x, y, yaw) in done:
            continue
        t0 = time.time()
        cmd = ['ros2', 'run', 'mycobot_tasks', 'tray_handle_pick', '--tray-type', 'test',
               '--layout-poses', '--auto', '--no-show', '--do', a.do,
               f'--relay-desk={x},{y},{yaw}'] + a.extra.split()
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout).stdout
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
            out += '\nTIMEOUT'
        m = re.search(r'relay plan desk -> TurtleBot OK in (\d+) s', out)
        pad = re.search(r'desk -> TurtleBot: planned with (\d+) mm clearance', out)
        far = 'far' if 'far desk tray' in out else ''
        why = ''
        if not m:
            lines = [l.strip() for l in out.splitlines()
                     if re.search(r'no way|ERROR|not on the desk|TIMEOUT|off the desk|Stop|reach', l)]
            why = ' | '.join(lines[-3:])[:300]
        w.writerow([x, y, yaw, int(bool(m)), m.group(1) if m else f'{time.time() - t0:.0f}',
                    pad.group(1) if pad else '', why, far])
        f.flush()
        print(f'({x:+.2f},{y:+.2f}) yaw {yaw:5.0f}: {"OK" if m else "--"} '
              f'{time.time() - t0:4.0f} s {why[:90]}', flush=True)
