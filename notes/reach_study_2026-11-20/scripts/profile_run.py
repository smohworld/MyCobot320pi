# tray_handle_pick with a planning profile: where the relay planning time goes. Wraps (no
# package change) the collision check (_contacts -> MoveIt check_state_validity), the IK
# (_ik_v / _ik_t), OMPL (_ompl), the carry grid (_grid), chains / turns / straighten and the
# dense check (relay_verify); exclusive times (a chain's own time without its IK / collision
# calls). After every relay plan (n part 1 / 2, j, e, r) one block "PROFILE ..." is printed.
# Same arguments as tray_handle_pick (simulation or real robot):
#   python3 profile_run.py --tray-type test --layout-poses --scene-args=--layout --do deskbot --no-show
import collections, sys, time
from mycobot_tasks.handle_pick import relay as R

T = collections.defaultdict(float)      # exclusive time per category [s]
N = collections.Counter()               # calls per category
stack = []                              # [category, start of its current own-time slice]


def enter(cat):
    now = time.perf_counter()
    if stack:
        c, t0 = stack[-1]
        T[c] += now - t0
    stack.append([cat, now])
    N[cat] += 1


def leave():
    now = time.perf_counter()
    c, t0 = stack.pop()
    T[c] += now - t0
    if stack:
        stack[-1][1] = now


def wrap(f, cat):
    def g(*a, **k):
        enter(cat)
        try:
            return f(*a, **k)
        finally:
            leave()
    return g


def wrap_gen(f, cat):
    def g(*a, **k):
        it = f(*a, **k)
        while True:
            enter(cat)
            try:
                v = next(it)
            except StopIteration:
                return
            finally:
                leave()
            yield v
    return g


def report(f, title):
    def g(self, *a, **k):
        top = not stack
        if top:
            T.clear()
            N.clear()
            t0 = time.perf_counter()
        enter('other')
        try:
            return f(self, *a, **k)
        finally:
            leave()
            if top:
                wall = time.perf_counter() - t0
                parts = sorted(T.items(), key=lambda kv: -kv[1])
                lines = [f'PROFILE {title}: {wall:.1f} s']
                for c, t in parts:
                    per = f', {t / N[c] * 1000:.1f} ms each' if N[c] else ''
                    lines.append(f'PROFILE   {c:12s} {t:6.1f} s {100 * t / max(wall, 1e-9):5.1f} %  '
                                 f'{N[c]:6d} calls{per}')
                print('\n'.join(lines), flush=True)
    return g


M = R.RelayMixin
for name, cat in (('_contacts', 'collision'), ('_ik_v', 'ik'), ('_ik_t', 'ik'),
                  ('_ompl', 'ompl'), ('_chain', 'chain'), ('_turn', 'turn'),
                  ('_straighten', 'straighten'), ('relay_verify', 'dense check')):
    setattr(M, name, wrap(getattr(M, name), cat))
M._grid = wrap_gen(M._grid, 'grid')
for name, title in (('relay_plan_a', 'D11 -> desk'), ('relay_plan_b', 'desk -> TurtleBot'),
                    ('relay_plan', 'D11 <-> TurtleBot'), ('relay_plan_desk', 'desk <-> TurtleBot')):
    setattr(M, name, report(getattr(M, name), title))
M.relay_verify = report(M.relay_verify, 'dense check')

from mycobot_tasks.handle_pick.cli import main   # noqa: E402
sys.argv = ['tray_handle_pick'] + sys.argv[1:]
main()
