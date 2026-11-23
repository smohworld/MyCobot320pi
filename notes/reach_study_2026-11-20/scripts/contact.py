import numpy as np, rclpy
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from mycobot_calibration.handeye_node import ARM_JOINTS
from mycobot_calibration.kin_calib import Arm, get_urdf
rclpy.init(); n = rclpy.create_node('ct'); c = n.create_client(GetStateValidity, 'check_state_validity'); c.wait_for_service(5)
u = get_urdf(None)
for qd in ([106, -39, -49, -2, 90, 106], [106, -85, 49, -54, 90, 106], [-107, 85, -49, 54, -90, 73]):
    q = np.radians(qd)
    rq = GetStateValidity.Request(); rq.group_name = 'arm_group'
    rq.robot_state = RobotState(joint_state=JointState(name=ARM_JOINTS, position=[float(v) for v in q]))
    f = c.call_async(rq); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
    print(qd, [(x.contact_body_1, x.contact_body_2, round(x.position.x, 3), round(x.position.y, 3), round(x.position.z, 3)) for x in f.result().contacts if 'desk' not in (x.contact_body_1 + x.contact_body_2)])
    print('   ', {l: np.round(Arm(u, tip=l).fk(q)[:3, 3], 3).tolist() for l in ('link1', 'link2', 'link3', 'link4')})
