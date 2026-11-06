"""myCobot 320 driver + robot_state_publisher (+ optional RViz).

  ros2 launch mycobot_bridge driver.launch.py                                   # fake
  ros2 launch mycobot_bridge driver.launch.py connection:=socket ip:=192.168.0.10
  ros2 launch mycobot_bridge driver.launch.py connection:=serial serial_port:=/dev/ttyUSB0

calibration_file:=<hand_eye.yaml> loads the OpenCV hand-eye result into the URDF
(link6 -> camera_optical_frame), so TF matches what MoveIt uses.
"""
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    desc = get_package_share_directory('mycobot_description')
    share = get_package_share_directory('mycobot_bridge')

    calib = lc('calibration_file')
    mappings = {'calibration_file': os.path.abspath(os.path.expanduser(calib))} if calib else {}
    xacro_file = os.path.join(desc, 'urdf', 'mycobot_320_pi_2022',
                              'pro_320_pi_2022_force_gripper.urdf.xacro')
    robot_description = xacro.process_file(xacro_file, mappings=mappings).toxml()

    actions = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen',
             parameters=[{'robot_description': ParameterValue(robot_description, value_type=str)}]),
        Node(package='mycobot_bridge', executable='mycobot_driver', name='mycobot_driver',
             output='screen',
             parameters=[os.path.join(share, 'config', 'driver.yaml'), {
                 'connection': lc('connection'),
                 'ip': lc('ip'),
                 'serial_port': lc('serial_port'),
                 'baudrate': int(lc('baudrate')),
                 'use_gripper': lc('use_gripper').lower() in ('true', '1'),
             }]),
    ]
    if lc('rviz').lower() in ('true', '1'):
        actions.append(Node(package='rviz2', executable='rviz2', output='log',
                            arguments=['-d', os.path.join(desc, 'rviz', 'display.rviz')]))
    return actions


def generate_launch_description():
    args = {
        'connection': 'fake',          # fake | socket | serial
        'ip': '192.168.0.10',
        'serial_port': '/dev/ttyUSB0',
        'baudrate': '115200',
        'use_gripper': 'true',
        'calibration_file': '',        # '' = mycobot_description/config/hand_eye.yaml
        'rviz': 'true',
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [OpaqueFunction(function=setup)])
