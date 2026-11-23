"""OpenCV window: camera image + overlay, keys -> actions."""
import time

import cv2
import numpy as np

from mycobot_tasks.red_cube_hover import CAM, TCP, project
from mycobot_tasks.tray_handle import MarkerPose, handle_from_markers
from mycobot_tasks.red_cube_hover import _text


def view(app):
    node, args = app.node, app.args
    win = 'tray_handle_pick'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    to_bot = app.wants_bot()      # m puts the tray onto the TurtleBot (--place-at bot)
    keys = {ord('p'): ('pick (any height)', app.act_pick),
            ord('m'): ('shelf -> TurtleBot' if to_bot else 'pick + place', app.act_pick_place),
            ord('k'): ('TurtleBot -> shelf', app.act_bot_to_shelf),
            ord('n'): ('D11 -> desk -> TurtleBot', app.act_d11_to_bot),
            ord('j'): ('TurtleBot -> desk -> D11', app.act_bot_to_d11),
            ord('d'): ('detect (any height)', app.act_detect),
            ord('s'): ('search pose', app.act_search),
            ord('b'): ('put back', app.act_place)}
    live = MarkerPose(app.K, app.D, args.marker_size)    # own detector (window thread)
    after = 0.0
    while True:
        img = node.fresh(after, timeout=0.1)
        if img is not None:
            after = time.monotonic()
            vis = img.copy()
            try:
                found = live.detect(img)
                for i, (c, rv, tv, _) in found.items():
                    mine = i in (app.id_l, app.id_r)
                    cv2.polylines(vis, [c.astype(np.int32)], True,
                                  (0, 255, 0) if mine else
                                  (255, 128, 0) if i == args.bot_id else (128, 128, 128), 2)
                    if mine:
                        cv2.drawFrameAxes(vis, app.K, app.D, rv, tv, args.marker_size * 0.6, 2)
                T_bc = node.T('base', CAM)
                if T_bc is not None:
                    r = handle_from_markers(found, app.id_l, app.id_r, T_bc, app.spacing)
                    if r is not None:
                        uv = project(r['p'], T_bc, app.K, app.D)
                        if uv:
                            cv2.drawMarker(vis, uv, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                        T_tcp = node.T('base', TCP)
                        if T_tcp is not None:
                            d = r['p'] - T_tcp[:3, 3]
                            _text(vis, f"handle: tip -> mid {np.linalg.norm(d) * 1000:.0f} mm "
                                       f"(dx {d[0] * 1000:+.0f} dy {d[1] * 1000:+.0f} dz "
                                       f"{d[2] * 1000:+.0f})  L-R {r['spacing'] * 1000:.1f} mm"
                                       f"{'' if r['n'] == 2 else ' (1 marker)'}",
                                  (10, vis.shape[0] - 36), 0.45)
                    for p, col in ((app.pin['p'] if app.pin else None, (0, 255, 255)),
                                   (app.target, (255, 0, 255))):
                        if p is not None:
                            uv = project(p, T_bc, app.K, app.D)
                            if uv:
                                cv2.drawMarker(vis, uv, col, cv2.MARKER_TILTED_CROSS, 18, 2)
                _text(vis, app.status, (10, 25), 0.55)
                _text(vis, f"{'' if args.go else 'DRY RUN  '}{'BUSY  ' if app.busy else ''}"
                           f"{'HOLDING  ' if app.holding else ''}{'AUTO' if app.auto else 'STEP'}",
                      (10, 50), 0.5)
                if app.prompt:
                    _text(vis, f'NEXT: {app.prompt}', (10, 78), 0.55)
                    _text(vis, 'Enter = go   x = cancel   a = auto', (10, 102), 0.55)
                _text(vis, ('b put back  ' if app.holding else '') + 'p pick  '
                           + ('m shelf->bot' if to_bot else 'm pick+place')
                           + '  k bot->shelf  n D11->bot  j bot->D11  d detect  s search  a auto  x stop  q quit',
                      (10, vis.shape[0] - 12), 0.45)
            except Exception as e:  # noqa: BLE001  (a drawing error must not end the app)
                if not getattr(app, '_draw_err', False):
                    print(f'window drawing error (ignored): {e!r}', flush=True)
                    app._draw_err = True
            cv2.imshow(win, vis)
        k = cv2.waitKey(10) & 0xFF
        if k in (ord('q'), 27):
            break
        if k in (ord('x'), ord(' ')):
            app.stop()
        elif k in (13, 10, 141):
            app._go_on.set()
        elif k == ord('a'):
            app.auto = not app.auto
            app.say('auto mode: no step questions' if app.auto else 'step mode: Enter per step')
            if app.auto:
                app._go_on.set()
        elif k in keys:
            app.submit(*keys[k])
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
    cv2.destroyWindow(win)
