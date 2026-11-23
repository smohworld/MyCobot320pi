# Whole route A (J6 within +-180):  shelf D11 (reversed arm) -> desk relay spot -> empty-hand
# arm switch (OMPL, tray on the desk as obstacle) -> regrasp turned 180 -> bot box (tray yaw 270).
# Saves the joint waypoints + scene/gripper steps to route_a.pkl for play_route.py.
#   python3 route_a.py [spot_x spot_y]
import math, sys, heapq, pickle, numpy as np
SPOT = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (-0.16, -0.14)
sys.argv = [sys.argv[0]]
from tilt2_hdr import ik_tilt, B, D11, TILT
from fastik import arm
from held import contacts_held, ik_local, ik_any, seg, chain
from scanlib import n, ap, _wait
from moveit_msgs.srv import GetMotionPlan, ApplyPlanningScene
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from mycobot_calibration.handeye_node import ARM_JOINTS
from scipy.spatial.transform import Rotation

DESK_TIP = -0.007 + 0.072 - 0.02; ZC = 0.125
Y0 = math.radians(270); Yc = Y0 + math.radians(90)          # tray yaw at D11 / on the desk
Yg = Yc + math.pi; Yb = math.radians(270)                   # regrasp turned 180, tray yaw in the box
OPEN, CLOSED = 0.5, 0.2
steps = []                                                  # (kind, data, label)
def traj(qs, label): steps.append(('traj', [np.asarray(q, float) for q in qs], label)); return qs[-1]
def fail(msg): print('FAILED:', msg); raise SystemExit(1)
def tray_pose(q):
    """world pose of the held tray box (TCP frame, centre 2.8 cm along TCP z)"""
    T = arm.fk(q); c = T[:3, 3] + T[:3, 2] * 0.028
    return c, Rotation.from_matrix(T[:3, :3]).as_quat()     # x y z w
def walk(q, pts, yaw, tilt=False, check=None, label=''):
    out = []
    for p in pts:
        qn = ik_tilt(p, yaw, q) if tilt else ik_local(p, yaw, q)
        if qn is None or np.degrees(np.max(np.abs(qn - q))) > 12: fail(f'{label}: no IK / jump at {np.round(p, 3)}')
        if check is not None and check(qn): fail(f'{label}: collision at {np.round(p, 3)} {check(qn)}')
        out.append(qn); q = qn
    return out
def chain_path(q, pts, yaw, skip=(), label=''):
    out = []
    for i, p in enumerate(pts):
        qe, w = chain(q, [p], yaw, skip=(0,) if i in skip else ())
        if qe is None: fail(f'{label}: {w}')
        out.append(qe); q = qe
    return out
def grid(S, qS, G, yaw, label):
    xs = np.round(np.arange(-0.30, 0.30, 0.02), 3); ys = np.round(np.arange(-0.30, 0.22, 0.02), 3)
    nodes = [(x, y) for x in xs for y in ys if math.hypot(x, y) >= 0.12] + [G]
    best = {S: 0.0}; qof = {S: qS}; par = {}; eq = {}; pq = [(0.0, S)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > best.get(u, 1e9) + 1e-9: continue
        if u == G: break
        for v in nodes:
            if v == u or math.dist(u, v) > 0.03 or d + math.dist(u, v) >= best.get(v, 1e9): continue
            qq = qof[u]; ok = True; e = []
            for p in seg([u[0], u[1], ZC], [v[0], v[1], ZC], step=0.005):
                qn = ik_tilt(p, yaw, qq)
                if qn is None or np.degrees(np.max(np.abs(qn - qq))) > 12: ok = False; break
                e.append(qn); qq = qn
            if not ok or contacts_held(qq): continue
            best[v] = d + math.dist(u, v); qof[v] = qq; par[v] = u; eq[v] = e; heapq.heappush(pq, (best[v], v))
    if G not in best: fail(f'{label}: no grid path ({len(best)} points)')
    path = []; v = G
    while v != S: path = eq[v] + path; v = par[v]
    print(f'{label}: {best[G]*100:.0f} cm over the grid', flush=True)
    return path
def straighten(q, p, yaw, label):
    out = []
    for t in np.linspace(0, 1, 21)[1:]:
        qn = ik_tilt(p, yaw, q, tilt=TILT * (1 - t) + 1e-3)
        if qn is None or np.degrees(np.max(np.abs(qn - q))) > 12: fail(f'{label}: straighten at t {t:.2f}')
        out.append(qn); q = qn
    qn = ik_local(p, yaw, q)
    if qn is None or np.degrees(np.max(np.abs(qn - q))) > 12: fail(f'{label}: no exact vertical')
    return out + [qn]

sx, sy = SPOT; Sd = np.array([sx, sy, DESK_TIP])
# ---- kinematics first (no extra world objects in the scene)
qD = ik_any(D11 - [0, 0, 0.02], Y0, True)
if qD is None: fail('no reversed grasp at D11')
app = walk(qD, seg(D11 - [0, 0, 0.02], D11 + [0, 0, 0.04]), Y0, label='D11 approach')[::-1][1:] + [qD]
qN = ik_any(Sd, Yg, False)
if qN is None: fail('no normal regrasp at the desk spot')
liftN = chain_path(qN, seg(Sd, [sx, sy, ZC]), Yg, skip=range(4), label='desk lift (normal)')
print('kinematic ends ok', flush=True)

# ---- leg A: D11 -> desk spot (reversed arm)
steps.append(('scene', ('add', 'shelf_tray_vis', tray_pose(qD)), 'tray on D11'))
q = traj(app, 'approach D11 (reversed arm)')
steps.append(('grip', CLOSED, 'close')); steps.append(('scene', ('attach', 'shelf_tray_vis'), 'attach'))
p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069])
q = traj(chain_path(q, seg(D11 - [0, 0, 0.02], D11 + [0, 0, 0.04]), Y0, skip=range(3), label='lift'), 'lift off D11')
q = traj(chain_path(q, seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2) + seg(p2, [p2[0], p2[1], ZC]), Y0, label='pull out'), 'pull out, up')
P = np.array([p2[0], p2[1], ZC])
turn = []
for t in np.linspace(0, 1, 30)[1:]:
    qn = ik_local(P, Y0 + (Yc - Y0) * t, q)
    if qn is None: fail('turn in place')
    turn.append(qn); q = qn
if contacts_held(q): fail('turn in place: collision')
q = traj(turn, 'turn tray +90 in place')
q = traj(grid((p2[0], p2[1]), q, (sx, sy), Yc, 'leg A grid'), 'carry (tilt <=10) to the desk spot')
q = traj(straighten(q, np.array([sx, sy, ZC]), Yc, 'leg A'), 'straighten')
q = traj(chain_path(q, seg([sx, sy, ZC], Sd + [0, 0, 0.018]), Yc, label='desk down'), 'down to the desk')
q = traj(walk(q, seg(Sd + [0, 0, 0.018], Sd, step=0.002), Yc, label='desk touch'), 'set down')   # tray bottom in the 1.2 cm desk margin: normal contact
qrel = q
steps.append(('grip', OPEN, 'open')); steps.append(('scene', ('detach', 'shelf_tray_vis'), 'tray stays on the desk'))
q = traj(walk(q, seg(Sd, [sx, sy, ZC]), Yc, label='empty up'), 'empty hand up')
qUR = q

# ---- arm switch reversed -> normal with the tray on the desk (OMPL)
c, quat = tray_pose(qrel)
def scene_box(op, name, c=None, quat=None):
    o = CollisionObject(); o.id = name; o.header.frame_id = 'base'; o.operation = op
    if op == CollisionObject.ADD:
        o.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.15, 0.10, 0.056])]
        pz = Pose(); pz.position.x, pz.position.y, pz.position.z = map(float, c)
        pz.orientation.x, pz.orientation.y, pz.orientation.z, pz.orientation.w = map(float, quat); o.primitive_poses = [pz]
    rq = ApplyPlanningScene.Request(scene=PlanningScene(is_diff=True)); rq.scene.world.collision_objects = [o]
    _wait(ap.call_async(rq), 10.0)
scene_box(CollisionObject.ADD, 'desk_tray_tmp', c + [0, 0, 0.004], quat)    # lifted 4 mm off the desk box
cli = n.create_client(GetMotionPlan, 'plan_kinematic_path'); cli.wait_for_service(5)
def ompl(qa, qb, label, tries=6):
    for k in range(tries):
        rq = GetMotionPlan.Request(); r = rq.motion_plan_request
        r.start_state = RobotState(joint_state=JointState(name=ARM_JOINTS + ['gripper_controller'], position=[float(v) for v in qa] + [OPEN]))
        r.group_name = 'arm'; r.num_planning_attempts = 5; r.allowed_planning_time = 10.0
        r.max_velocity_scaling_factor = 0.3; r.max_acceleration_scaling_factor = 0.3
        r.goal_constraints = [Constraints(joint_constraints=[JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005, tolerance_below=0.005, weight=1.0) for j, v in zip(ARM_JOINTS, qb)])]
        res = _wait(cli.call_async(rq), 40.0).motion_plan_response
        if res.error_code.val != 1: print(f'   {label}: plan {k} failed {res.error_code.val}'); continue
        jt = res.trajectory.joint_trajectory; idx = [list(jt.joint_names).index(j) for j in ARM_JOINTS]
        pts = [np.array(pt.positions)[idx] for pt in jt.points]
        dense = []                                           # check every <=2 deg
        for a, b in zip(pts, pts[1:]):
            m = max(1, int(np.degrees(np.max(np.abs(b - a))) / 2))
            dense += [a + (b - a) * t for t in np.linspace(0, 1, m + 1)[1:]]
        bad = [i for i, qq in enumerate(dense) if contacts_held(qq, OPEN, False)]
        if bad: print(f'   {label}: plan {k} has {len(bad)} colliding substeps'); continue
        print(f'{label}: {len(pts)} points, J1 {np.degrees(qa[0]):.0f} -> {np.degrees(qb[0]):.0f}', flush=True)
        return dense
    fail(f'{label}: no clean plan')
q = traj(ompl(qUR, liftN[-1], 'arm switch reversed -> normal'), 'arm switch (empty, over the tray)')
scene_box(CollisionObject.REMOVE, 'desk_tray_tmp')

# ---- leg B: desk spot -> bot box (normal arm, tray yaw 270)
q = traj(liftN[::-1][1:] + [qN], 'down to the tray (regrasp turned 180)')
steps.append(('grip', CLOSED, 'close')); steps.append(('scene', ('attach', 'shelf_tray_vis'), 'attach'))
q = traj(liftN, 'lift off the desk')
d = math.remainder(Yb - Yg, 2 * math.pi); turn = []
for t in np.linspace(0, 1, 30)[1:]:
    qn = ik_tilt(np.array([sx, sy, ZC]), Yg + d * t, q)
    if qn is None: fail('turn for the box')
    turn.append(qn); q = qn
if contacts_held(q): fail('turn for the box: collision')
q = traj(turn, f'turn tray {math.degrees(d):+.0f}')
G = (float(B[0]), float(B[1]))
q = traj(grid((sx, sy), q, G, Yb, 'leg B grid'), 'carry (tilt <=10) over the bot')
q = traj(walk(q, seg([G[0], G[1], ZC], B + [0, 0, 0.08]), Yb, tilt=True, check=contacts_held, label='tilted descent'), 'tilted descent to +8 cm')
q = traj(straighten(q, B + [0, 0, 0.08], Yb, 'leg B'), 'straighten')
q = traj(chain_path(q, seg(B + [0, 0, 0.08], B), Yb, label='into the box'), 'down into the box')
steps.append(('grip', OPEN, 'open')); steps.append(('scene', ('detach', 'shelf_tray_vis'), 'tray in the box'))
q = traj(walk(q, seg(B, B + [0, 0, 0.06]), Yb, label='leave'), 'empty hand up')
print(f'J6 range over the route: {min(np.degrees(qq[5]) for k, s, _ in steps if k == "traj" for qq in s):.0f} .. '
      f'{max(np.degrees(qq[5]) for k, s, _ in steps if k == "traj" for qq in s):.0f} deg')
pickle.dump(dict(spot=SPOT, steps=steps, tray_dims=[0.15, 0.10, 0.056]), open('route_a.pkl', 'wb'))
print('saved route_a.pkl:', sum(len(s) for k, s, _ in steps if k == 'traj'), 'waypoints,',
      len([1 for k, *_ in steps if k == 'traj']), 'segments')
