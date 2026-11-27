# n / j dry runs (plan + dense check + servo-error check, nothing moves) at the Burger poses
# measured on the real robot (tray_handle_pick.log, 11-26 / 11-27), with real errors put in:
#   --sim-desk-error   the tray lands off the plan on the desk (real r: up to 4.3 mm / 1.8 deg)
#   --sim-d11-error    the D11 pose (reference) is off the shelf (measurement 2-3 mm)
#   PERTURB_DEG        every dense pose again with the joints off by +-D (real held off-path
#                      max 0.9 deg with ff_along_path) -> perturb_run.py
# Each case: layout_bot / layout_tray_bot / layout.yaml bot in $HOME/.ros rewritten (sim home!).
#   ROS_DOMAIN_ID=77 HOME=<sim home> python3 njreal.py OUT.csv LOGDIR [--modes n,j] [--extra ...]
import argparse, csv, math, os, re, subprocess, sys, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# real Burger box marker poses (x, y [m], yaw [deg]) from tray_handle_pick.log, 11-26 / 11-27:
# the yaw as the Burger heading marker ID 28 confirmed it (r / j: from tray + ID28; e / n: ID0,
# ID28 agrees), so no 180 deg guess here
POSES = {
    'B01_1439': (0.261, -0.017, -36), 'B02_1447': (0.272, -0.010, 20),
    'B03_1457': (0.254, -0.029, -172), 'B04_1510': (0.216, -0.134, -84),
    'B05_1535': (0.227, -0.118, -93), 'B06_1639': (0.246, -0.124, -93),
    'B07_1721': (0.246, -0.098, 173), 'B08_1726': (0.227, -0.124, -132),
    'B09_1732': (0.242, -0.115, -143), 'B10_0928': (0.220, -0.077, -95),
    'B11_0930': (0.238, -0.060, -99), 'B12_0933': (0.279, -0.076, -90),
}
Z = -0.277          # box marker z (real -274 .. -279 mm)

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('logdir')
ap.add_argument('--modes', default='n,j')
ap.add_argument('--poses', default='', help='comma list of pose names (default all)')
ap.add_argument('--tag', default='', help='label column (e.g. pull20)')
ap.add_argument('--extra', default='', help='more tray_handle_pick options')
ap.add_argument('--timeout', type=float, default=900)
ap.add_argument('--scene', default='--layout', help='place_scene args (e.g. "--layout --rack-dy 0.015")')
ap.add_argument('--fork-top', type=float, default=None,
                help='D11 fork top over the floor [m] (layout.yaml fork_top + layout_tray.yaml '
                     'handle z moved alike; restored at the end). Model 0.263, real ~0.24?')
a = ap.parse_args()

ros = os.path.expanduser('~/.ros')
tray_h = 0.078          # shelf tray handle top (real 78 mm, 11-27)


with open(f'{ros}/layout.yaml') as _f:
    LAY0 = _f.read()
with open(f'{ros}/layout_tray.yaml') as _f:
    TRAY0 = _f.read()
FORK0 = yaml.safe_load(LAY0)['fork_top']


def write_fork():
    if a.fork_top is None:
        return
    t = yaml.safe_load(TRAY0)
    dz = a.fork_top - FORK0
    t['handle_xyz'][2] = round(t['handle_xyz'][2] + dz, 4)
    t['table_z'] = round(t['table_z'] + dz, 4)
    with open(f'{ros}/layout_tray.yaml', 'w') as f:
        yaml.safe_dump(t, f)


def write_layout(x, y, yaw):
    st = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(f'{ros}/layout_bot.yaml', 'w') as f:
        yaml.safe_dump(dict(stamp=st, marker_id=0, marker_xyz=[x, y, Z], yaw_deg=float(yaw),
                            tilt_deg=0.0), f)
    with open(f'{ros}/layout_tray_bot.yaml', 'w') as f:
        yaml.safe_dump(dict(stamp=st, tray=0, handle_xyz=[x, y, Z + tray_h],
                            yaw_lr_deg=float(yaw), spacing=0.057, tilt_deg=0.0,
                            table_z=Z, camera_q_deg=None), f)
    with open(f'{ros}/layout.yaml') as f:
        lay = yaml.safe_load(f)
    lay['bot'], lay['with_bot'] = [x, y, float(yaw)], True
    if a.fork_top is not None:
        lay['fork_top'] = a.fork_top
    with open(f'{ros}/layout.yaml', 'w') as f:
        yaml.safe_dump(lay, f, sort_keys=False)


def grab(pat, out, g=1):
    m = re.findall(pat, out)
    return m[-1] if m else ''


names = [n for n in POSES if not a.poses or n in a.poses.split(',')]
new = not os.path.exists(a.out)
f = open(a.out, 'a', newline='')
w = csv.writer(f)
if new:
    w.writerow(['tag', 'fork_top', 'pose', 'x', 'y', 'yaw', 'mode', 'ok', 'run_s', 'spot', 'plan1_s',
                'pad1_mm', 'plan2_s', 'pad2_mm', 'dense', 'perturb', 'near', 'why', 'log'])
os.makedirs(a.logdir, exist_ok=True)
write_fork()
try:
  for name in names:
      x, y, yaw = POSES[name]
      write_layout(x, y, yaw)
      for mode in a.modes.split(','):
          do = {'n': 'd11bot', 'j': 'botd11', 'e': 'deskbot', 'r': 'botdesk'}[mode]
          log = os.path.join(a.logdir, f'{a.tag or "run"}_{name}_{mode}.log')
          cmd = [sys.executable, os.path.join(HERE, 'perturb_run.py'), '--layout-poses',
                 f'--scene-args={a.scene}', '--auto', '--no-show', '--do', do] + a.extra.split()
          t0 = time.time()
          try:
              out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout,
                                   stdin=subprocess.DEVNULL).stdout
          except subprocess.TimeoutExpired as e:
              out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')) + \
                  '\nTIMEOUT'
          dt = time.time() - t0
          open(log, 'w').write(out)
          end = {'n': 'tray on the TurtleBot (dry run)', 'j': 'tray on D11 (dry run)',
                 'e': 'tray on the TurtleBot (dry run)', 'r': 'tray on the desk (dry run)'}[mode]
          ok = (re.search(r'tray on the desk( at \([^)]*\) mm)? \(dry run\)', out) is not None
                if mode == 'r' else end in out) and 'ERROR' not in out
          if mode == 'n':
              spot = grab(r'relay plan D11 -> desk OK in \d+ s: desk spot \(([^)]*)\)', out)
              p1 = grab(r'relay plan D11 -> desk OK in (\d+) s', out)
              p2 = grab(r'relay plan desk -> TurtleBot OK in (\d+) s', out)
              pad1 = grab(r'D11 -> desk: planned with (\d+) mm', out)
              pad2 = grab(r'desk -> TurtleBot: planned with (\d+) mm', out)
          else:
              spot = grab(r'relay plan OK in \d+ s: desk spot \(([^)]*)\)', out)
              p1 = grab(r'relay plan OK in (\d+) s', out)
              p2 = grab(r'relay plan desk -> D11 OK in (\d+) s', out)
              pad1 = grab(r'D11 <-> TurtleBot: planned with (\d+) mm', out)
              pad2 = grab(r'desk -> D11: planned with (\d+) mm', out)
          if 'NO clearance margin' in out:
              pad1 = pad1 or '0'
          dense = 'COLLISION' if re.search(r'^(?!PERTURB).*COLLISION in the dense check', out, re.M) \
              else ('OK' if 'dense collision check OK' in out else '')
          pert_ok = len(re.findall(r'^PERTURB [^\n]*dense collision check OK', out, re.M))
          pert_hit = re.findall(r'^PERTURB [^\n]*HIT:?\n((?:  [^\n]*\n?)*)', out, re.M)
          hits = [f'{d} deg {t}' for d, t in re.findall(r'PERTURB ([\d.]+) deg ([^:\n]*): HIT', out)]
          perturb = f'{pert_ok} ok' + (f', HIT in {"; ".join(hits)}' if hits else '')
          hl = [l.strip() for l in out.splitlines() if l.startswith('  ') and ' of ' in l and
                'poses' in l]
          near = len(re.findall(r'closer than \d+ mm in (\d+) paths', out))
          why = ''
          if not ok:
              L = [l.strip() for l in out.splitlines() if re.search(
                  r'ERROR|no way|failed|TIMEOUT|Traceback|Error|Stop', l)]
              why = ' | '.join(L[-4:])[:400]
          if hits:
              why = (why + ' | ' if why else '') + ' / '.join(hl[:3])[:300]
          w.writerow([a.tag, a.fork_top if a.fork_top is not None else FORK0, name, x, y, yaw, mode, int(ok), f'{dt:.0f}', spot, p1, pad1, p2,
                      pad2, dense, perturb, near, why, log])
          f.flush()
          print(f'{name} {mode}: {"OK" if ok else "FAIL"} {dt:4.0f} s spot ({spot}) plan '
                f'{p1}+{p2} s pad {pad1}/{pad2} dense {dense} perturb {perturb} {why[:120]}',
                flush=True)
finally:
    with open(f'{ros}/layout.yaml', 'w') as _f:
        _f.write(LAY0)
    with open(f'{ros}/layout_tray.yaml', 'w') as _f:
        _f.write(TRAY0)
