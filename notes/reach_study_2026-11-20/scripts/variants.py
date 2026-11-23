# Layout variants for the relay (n / j) without touching ~/.ros: each variant = the current
# ~/.ros/layout.yaml with one change, saved + exported into OUT/<name>/, then
# tray_handle_pick --layout-poses plans it (dry run, own sim ROS_DOMAIN_ID).
#   python3 variants.py OUT [d11bot|botd11 ...]
import os, re, subprocess, sys, time
from mycobot_tasks import layout as lay

OUT = os.path.abspath(sys.argv[1]); DOS = sys.argv[2:] or ['d11bot']


def bot(dx=0.0, dy=0.0, dyaw=0.0):
    def f(v):
        v['bot'] = [v['bot'][0] + dx, v['bot'][1] + dy, v['bot'][2] + dyaw]
    return f


def key(k, val):
    def f(v):
        v[k] = val
    return f


def shelf(dx=0.0, dy=0.0):
    def f(v):
        v['shelf'] = [v['shelf'][0] + dx, v['shelf'][1] + dy, v['shelf'][2]]
    return f


VARIANTS = dict(base=lambda v: None,
                bot_x_p3=bot(dx=0.03), bot_x_m3=bot(dx=-0.03),
                bot_y_p3=bot(dy=0.03), bot_y_m3=bot(dy=-0.03),
                bot_yaw_p30=bot(dyaw=30), bot_yaw_m30=bot(dyaw=-30), bot_yaw_p90=bot(dyaw=90),
                fork_287=key('fork_top', 0.287), shelf_x_p2=shelf(dx=0.02),
                shelf_y_m2=shelf(dy=-0.02),
                face_shelf=None)
only = os.environ.get('ONLY')
for name, mod in VARIANTS.items():
    if only and name not in only.split(','):
        continue
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    L = lay.Layout('~/.ros/layout.yaml')
    if name == 'face_shelf':
        L.turn_world(-90.0)       # robot +x toward the shelf: desk / bot / shelf turned -90 about the base
    else:
        mod(L.v)
    L.path = os.path.join(d, 'layout.yaml')
    L.save()
    lay.FILES = {k: os.path.join(d, os.path.basename(p)) for k, p in lay.FILES.items()}
    L.export()
    for do in DOS:
        t0 = time.time()
        r = subprocess.run(['ros2', 'run', 'mycobot_tasks', 'tray_handle_pick', '--layout-poses',
                            '--relay-plan-pad', '0.005', '--bot-yaw-offset', '90',
                            '--layout-dir', d, f'--scene-args=--layout {L.path}', '--do', do,
                            '--no-show', '--log', ''], capture_output=True, text=True, timeout=1500)
        out = r.stdout + r.stderr
        open(os.path.join(d, f'{do}.log'), 'w').write(out)
        ok = re.search(r'relay plan OK.*', out)
        summ = re.search(r'(D11 -> TurtleBot|TurtleBot -> D11): .*J6.*', out)
        if ok:
            res = f"OK   {ok.group(0).split(': ', 1)[-1]}; {summ.group(0) if summ else ''}"
        else:
            m = re.search(r'relay plan failed.*?(?=\n\S|\Z)', out, re.S)
            res = 'FAIL ' + (m.group(0).replace('\n', ' | ')[:600] if m else out[-400:])
        print(f'{name:12s} {do:7s} {time.time() - t0:4.0f} s  bot {[round(x, 3) for x in L.v["bot"]]}'
              f'  {res}', flush=True)
