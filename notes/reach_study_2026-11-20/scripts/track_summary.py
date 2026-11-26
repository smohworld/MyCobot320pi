# track_log.csv (mycobot_bridge driver, one row per trajectory) per relay move:
# how far the real joints were OFF THE PLANNED PATH while moving (nearest path point,
# so time lag along the path does not count).
#   off_j  = the largest single joint at the worst point = what perturb_run.py checks
#            (every joint +-0.5 / 1.0 deg): <= 0.5 survives in sim, 1.0 hits in places
#   lag_j  = planned(t) - actual(t), largest joint (includes the ~0.12 s lag; info only)
# Held moves (tray in the gripper) are what matters; empty moves are listed for context.
#   python3 track_summary.py [CSV] [--all] [--since HH:MM | unix] [--until HH:MM | unix]
#                            [--warn 0.5] [--rows] [--by-speed]
#   --by-speed: held moves per move name AND planned top joint speed (vmax, rounded to
#   0.05 rad/s), side by side: for the --relay-held-speed A/B test (e.g. 0.35 vs 0.20)
#   --by ff|lookahead|rate|stream|speed: the same per driver setting (columns written by the
#   driver since 11-26: ff_path, lookahead, loop_hz, stream_speed; older rows '-'), with the
#   achieved loop rate (tick_hz) of each group. Several CSVs can be given (the driver moves
#   an older-column file to track_log.csv.vN): default track_log.csv + its .vN copies.
import argparse, csv, math, os, re, time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('csv', nargs='*', default=None)
ap.add_argument('--all', action='store_true', help='include fake (sim) rows')
ap.add_argument('--since', help='HH:MM today or unix time')
ap.add_argument('--warn', type=float, default=0.5, help='off_j warning [deg]')
ap.add_argument('--until', help='HH:MM today or unix time')
ap.add_argument('--rows', action='store_true', help='every move, not only the summary')
ap.add_argument('--by-speed', action='store_true',
                help='held moves per name and planned top joint speed (A/B test)')
ap.add_argument('--by', choices=['ff', 'lookahead', 'rate', 'stream', 'speed'],
                help='held moves per name and this driver setting (A/B test)')
a = ap.parse_args()


def when(v, default):
    if not v:
        return default
    if ':' in v:
        h, m = map(int, v.split(':'))
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))
    return float(v)


since, until = when(a.since, 0.0), when(a.until, math.inf)

import glob
files = a.csv or ([os.path.expanduser('~/final_project_ws/track_log.csv')] + sorted(
    glob.glob(os.path.expanduser('~/final_project_ws/track_log.csv.v*'))))
rows = []
for fn in files:
  if not os.path.exists(os.path.expanduser(fn)):
    continue
  with open(os.path.expanduser(fn)) as f:
    for r in csv.DictReader(f):
        if 'off1' not in r or 'label' not in r or 'vmax1' not in r:
            continue
        if not a.all and r['connection'] == 'fake':
            continue
        if not since <= float(r['time']) < until:
            continue
        r['off_j'] = max(float(r[f'off{j}']) for j in range(1, 7))
        r['off_k'] = 1 + max(range(6), key=lambda j: float(r[f'off{j + 1}']))
        r['lag_j'] = max(float(r[f'max{j}']) for j in range(1, 7))
        # planned top joint speed of this move (vmax columns: deg/s) -> rad/s, 0.05 steps
        r['spd'] = round(math.radians(max(float(r[f'vmax{j}']) for j in range(1, 7))) / 0.05) * 0.05
        r['key'] = re.sub(r'[-+]?\d+(\.\d+)?', '#', r.get('label') or '(no label)')
        rows.append(r)
rows.sort(key=lambda r: float(r['time']))

if not rows:
    raise SystemExit('no rows (real rows only; --all for sim, check --since)')

t0, t1 = float(rows[0]['time']), float(rows[-1]['time'])
print(f'{len(rows)} moves, {time.strftime("%m-%d %H:%M", time.localtime(t0))} .. '
      f'{time.strftime("%H:%M", time.localtime(t1))}, warn off_j > {a.warn} deg (* = held)')

if a.rows:
    print(f'\n{"time":8} {"mode":4} {"h":1} {"dur":>5} {"off_j":>5} {"J":>2} {"lag_j":>5}  label')
    for r in rows:
        flag = '!' if r['held'] == '1' and r['off_j'] > a.warn else ' '
        print(f'{time.strftime("%H:%M:%S", time.localtime(float(r["time"])))} '
              f'{r["mode"]:4} {r["held"]:1} {float(r["duration_s"]):5.1f} {r["off_j"]:5.2f} '
              f'J{r["off_k"]} {r["lag_j"]:5.2f} {flag}{r["label"]}')

g = defaultdict(list)
for r in rows:
    g[(r['held'] == '1', r['key'])].append(r)
print(f'\n{"":1} {"n":>3} {"off_j max":>9} {"mean":>5} {"J":>3} {"lag_j":>5}  move')
for (held, key), rs in sorted(g.items(), key=lambda kv: (not kv[0][0], -max(r['off_j'] for r in kv[1]))):
    w = max(rs, key=lambda r: r['off_j'])
    mean = sum(r['off_j'] for r in rs) / len(rs)
    flag = '!' if held and w['off_j'] > a.warn else ' '
    print(f'{"*" if held else " "} {len(rs):3} {w["off_j"]:9.2f} {mean:5.2f} J{w["off_k"]:<2} '
          f'{max(r["lag_j"] for r in rs):5.2f} {flag}{key}')

held = [r for r in rows if r['held'] == '1']
if held:
    bad = [r for r in held if r['off_j'] > a.warn]
    w = max(held, key=lambda r: r['off_j'])
    print(f'\nheld: {len(held)} moves, worst off_j {w["off_j"]:.2f} deg (J{w["off_k"]}, '
          f'{w["label"]}), {len(bad)} over {a.warn}')

if a.by_speed:
    hs = [r for r in rows if r['held'] == '1']
    spds = sorted({r['spd'] for r in hs})
    if not hs:
        print('\n--by-speed: no held moves')
    else:
        print(f'\n--by-speed: held off_j mean / max [deg] (n) per planned top joint speed '
              f'[rad/s]; same move, lower speed lower off_j = speed matters')
        print(f'{"move":44} ' + ' '.join(f'{s_:>16.2f}' for s_ in spds))
        keys = sorted({r['key'] for r in hs})
        for k in keys + ['(all held)']:
            cells = []
            for s_ in spds:
                rs = [r for r in hs if r['spd'] == s_ and (k == '(all held)' or r['key'] == k)]
                cells.append(f'{sum(r["off_j"] for r in rs) / len(rs):5.2f} /{max(r["off_j"] for r in rs):5.2f} ({len(rs):2})'
                             if rs else f'{"-":>16}')
            print(f'{k[:44]:44} ' + ' '.join(f'{c:>16}' for c in cells))

if a.by_speed and not a.by:
    a.by = 'speed'
if a.by and a.by != 'speed' or (a.by == 'speed' and not a.by_speed):
    col = dict(ff='ff_path', lookahead='lookahead', rate='loop_hz', stream='stream_speed')
    def val(r):
        if a.by == 'speed':
            return f"{r['spd']:.2f}"
        v = r.get(col[a.by])
        return '-' if v in (None, '') else v
    hs = [r for r in rows if r['held'] == '1']
    vals = sorted({val(r) for r in hs})
    print(f'\n--by {a.by}: held off_j mean / max [deg] (n) per setting; achieved loop rate '
          f'tick_hz below (set loop_hz may not be reached)')
    print(f'{"move":44} ' + ' '.join(f'{v:>16}' for v in vals))
    for k in sorted({r['key'] for r in hs}) + ['(all held)']:
        cells = []
        for v in vals:
            rs = [r for r in hs if val(r) == v and (k == '(all held)' or r['key'] == k)]
            cells.append(f'{sum(r["off_j"] for r in rs) / len(rs):5.2f} /{max(r["off_j"] for r in rs):5.2f} ({len(rs):2})'
                         if rs else '-')
        print(f'{k[:44]:44} ' + ' '.join(f'{c:>16}' for c in cells))
    cells = []
    for v in vals:
        t = [float(r['tick_hz']) for r in hs if val(r) == v and r.get('tick_hz')]
        cells.append(f'{sum(t) / len(t):.1f} Hz' if t else '-')
    print(f'{"achieved loop rate (tick_hz, mean)":44} ' + ' '.join(f'{c:>16}' for c in cells))
