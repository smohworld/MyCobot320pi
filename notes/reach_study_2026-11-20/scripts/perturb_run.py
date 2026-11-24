# tray_handle_pick with an extra "servo error" check after every relay dense check: every
# dense pose again with the joints off by +-D deg (J1..J5 one at a time, + 6 random mixes of
# all six within +-D), D from PERTURB_DEG (default "0.5,1.0"). Report only (never stops);
# lines start with "PERTURB". Nothing in the package is changed (monkeypatch here).
#   ROS_DOMAIN_ID=77 python3 perturb_run.py <tray_handle_pick args>
import math, os, sys
import numpy as np
from mycobot_tasks.handle_pick import relay as R
from mycobot_tasks.red_cube_hover import Stop

DEGS = [float(v) for v in os.environ.get('PERTURB_DEG', '0.5,1.0').split(',')]
orig_verify, orig_contacts = R.RelayMixin.relay_verify, R.RelayMixin._contacts


def offsets(deg):
    d = math.radians(deg)
    out = []
    for j in range(5):
        for s in (1, -1):
            e = np.zeros(6)
            e[j] = s * d
            out.append(e)
    rng = np.random.default_rng(7)
    out += [rng.uniform(-d, d, 6) for _ in range(6)]
    return out


def contacts(self, q, *a, **k):
    pert = getattr(self, '_pert', None)
    if not pert:
        return orig_contacts(self, q, *a, **k)
    hit = set()
    for e in pert:
        hit |= set(orig_contacts(self, np.asarray(q, float) + e, *a, **k))
    return sorted(hit)


def verify(self, steps, title, start_tray=True):
    near = orig_verify(self, steps, title, start_tray)
    keep = self.args.relay_pad
    for deg in DEGS:
        self._pert, self.args.relay_pad = offsets(deg), 0.0
        try:
            orig_verify(self, steps, f'PERTURB {deg:g} deg {title}', start_tray)
        except Stop as e:
            print(str(e).replace('COLLISION in the dense check (nothing moved)', 'HIT'), flush=True)
        finally:
            self._pert, self.args.relay_pad = None, keep
    return near


R.RelayMixin.relay_verify, R.RelayMixin._contacts = verify, contacts
from mycobot_tasks.handle_pick.cli import main   # noqa: E402
sys.argv = ['tray_handle_pick'] + sys.argv[1:]
main()
