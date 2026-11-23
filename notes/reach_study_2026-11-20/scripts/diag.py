import math, numpy as np, rclpy
from fastik import solve, arm
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from mycobot_calibration.handeye_node import ARM_JOINTS
rclpy.init(); n = rclpy.create_node('diag'); cli = n.create_client(GetStateValidity, 'check_state_validity'); cli.wait_for_service(5)
def check(q):
    rq = GetStateValidity.Request(); rq.group_name = 'arm_group'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q]))
    f = cli.call_async(rq); rclpy.spin_until_future_complete(n, f, timeout_sec=5); r = f.result()
    return r.valid, sorted({(c.contact_body_1, c.contact_body_2) for c in r.contacts})
if __name__ == "__main__":
    hz = -0.306 + 0.263 + 0.072
    for h in (0.04, 0.02, -0.02):
        for trial in range(4):
            q = solve(np.array([0.003, 0.313, hz + h]), math.pi)
            if q is None: print(h, 'no ik'); break
            print(f'h {h:+.2f} q(deg) {np.round(np.degrees(q)).astype(int)}', check(q))
