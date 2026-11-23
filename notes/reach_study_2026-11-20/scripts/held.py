import math, numpy as np
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState, AttachedCollisionObject, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from scanlib import sv, _wait
from mycobot_calibration.handeye_node import ARM_JOINTS
from route import ik_local, ik_any
TOUCH = ['gripper_connection', 'gripper_base', 'gripper_left1', 'gripper_left2', 'gripper_left3',
         'gripper_right1', 'gripper_right2', 'gripper_right3', 'gripper_tcp']
def tray_obj():
    a = AttachedCollisionObject(); a.link_name = 'gripper_tcp'; a.touch_links = TOUCH
    o = a.object; o.id = 'held_tray'; o.header.frame_id = 'gripper_tcp'; o.operation = CollisionObject.ADD
    b = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.15, 0.10, 0.056])   # TCP x across the bar = tray length
    p = Pose(); p.position.z = 0.028; p.orientation.w = 1.0                       # TCP z down: body from tips to 5.6 cm below
    o.primitives = [b]; o.primitive_poses = [p]
    return a
TRAY = tray_obj()
def contacts_held(q, grip=0.2, held=True):
    rq = GetStateValidity.Request(); rq.group_name = 'arm'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS + ['gripper_controller'], position=[float(v) for v in q] + [grip]),
                                attached_collision_objects=[TRAY] if held else [], is_diff=False)
    r = _wait(sv.call_async(rq), 20.0)
    return sorted({tuple(sorted((c.contact_body_1, c.contact_body_2))) for c in r.contacts} - {('base', 'desk')})
def seg(a, b, polar=False, step=0.005):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if polar:
        ra, rb = math.hypot(*a[:2]), math.hypot(*b[:2]); ba, bb = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
        d = math.remainder(bb - ba, 2 * math.pi); n = max(2, int((abs(d) * max(ra, rb) + abs(rb - ra) + abs(b[2] - a[2])) / step))
        return [np.array([(ra + (rb - ra) * t) * math.cos(ba + d * t), (ra + (rb - ra) * t) * math.sin(ba + d * t), a[2] + (b[2] - a[2]) * t]) for t in np.linspace(0, 1, n)[1:]]
    n = max(2, int(np.linalg.norm(b - a) / step)); return [a + (b - a) * t for t in np.linspace(0, 1, n)[1:]]
def chain(q, pts, yaw, skip=()):
    for i, p in enumerate(pts):
        qn = ik_local(p, yaw, q)
        if qn is None: return None, f'no IK at {np.round(p, 3)}'
        if np.degrees(np.max(np.abs(qn - q))) > 12: return None, f'branch jump at {np.round(p, 3)}'
        if (i % 2 == 0 or i == len(pts) - 1) and i not in skip:
            c = contacts_held(qn)
            if c: return None, f'collision at {np.round(p, 3)} {c}'
        q = qn
    return q, 'ok'
D11 = np.array([0.003, 0.313, 0.029]); B = np.array([0.2155, -0.0449, -0.208])
if __name__ == '__main__':
    for yd in (270, 90):
        Y = math.radians(yd)
        q0 = ik_any(D11 - [0, 0, 0.02], Y, True)
        if q0 is None: print(yd, 'no reversed grasp'); continue
        print(f'tool yaw {yd}: grasp at D11 (reversed) q {np.round(np.degrees(q0)).astype(int)}; open-gripper contacts {contacts_held(q0, 0.5, False)}')
        lift = seg(D11 - [0, 0, 0.02], D11 + [0, 0, 0.04])
        qa, w = chain(q0, lift, Y, skip=range(3))                      # first ~1.5 cm: tray still on the fork
        print('   lift off D11 (tray held):', w, flush=True)
        if qa is None: continue
        p1 = D11 + [0.02, 0, 0.04]; p2 = np.array([0.023, 0.16, 0.069])
        qb, w = chain(qa, seg(D11 + [0, 0, 0.04], p1) + seg(p1, p2), Y)
        print('   +x 2 cm, pull to y 0.16 (tray held):', w, flush=True)
        if qb is None: continue
        for zc in (0.125, 0.18, 0.22):
            p3 = np.array([p2[0], p2[1], zc]); qc, w = chain(qb, seg(p2, p3), Y)
            if qc is None: print(f'   up to z {zc}: {w}'); continue
            for mid in ((0.10, 0.15), (0.15, 0.12), (0.20, 0.05)):
                M = np.array([mid[0], mid[1], zc])
                q, w = chain(qc, seg(p3, M) + seg(M, [B[0], B[1], zc], polar=True) + seg([B[0], B[1], zc], B + [0, 0, 0.04]), Y)
                print(f'   z {zc}: via ({mid[0]:+.2f},{mid[1]:+.2f}) -> bot +4 cm: {w}', flush=True)
                if q is not None:
                    q2, w2 = chain(q, seg(B + [0, 0, 0.04], B + [0, 0, 0.0]), Y, skip=())
                    print('      down to the bot (tips at the handle top):', w2)
