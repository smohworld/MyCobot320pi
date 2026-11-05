"""myCobot 320 + MoveIt2 with the real mycobot_bridge driver (no ros2_control).

  ros2 launch mycobot_moveit_config bringup.launch.py connection:=fake          # no hardware
  ros2 launch mycobot_moveit_config bringup.launch.py ip:=192.168.0.10          # socket
  ros2 launch mycobot_moveit_config bringup.launch.py connection:=serial serial_port:=/dev/ttyUSB0
  hand-eye result: calibration_file:=~/handeye_calib/<run>/hand_eye.yaml

The driver launch starts robot_state_publisher with the same xacro + calibration_file, so
move_group and TF use the same model (camera_optical_frame = hand-eye result).
MoveIt talks to the driver through moveit_simple_controller_manager:
  arm_controller/follow_joint_trajectory, gripper_controller/gripper_cmd
(demo.launch.py from the Setup Assistant needs ros2_control mock hardware and is not used.)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

ROBOT = 'mycobot_320_pi_2022_force_gripper'


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    desc = get_package_share_directory('mycobot_description')
    bridge = get_package_share_directory('mycobot_bridge')
    share = get_package_share_directory('mycobot_moveit_config')

    calib = lc('calibration_file')
    calib = os.path.abspath(os.path.expanduser(calib)) if calib else ''
    mappings = {'calibration_file': calib} if calib else {}
    xacro_file = os.path.join(desc, 'urdf', 'mycobot_320_pi_2022',
                              'pro_320_pi_2022_force_gripper.urdf.xacro')

    cfg = (MoveItConfigsBuilder(ROBOT, package_name='mycobot_moveit_config')
           .robot_description(file_path=xacro_file, mappings=mappings)
           .trajectory_execution(file_path='config/moveit_controllers.yaml')
           .joint_limits(file_path='config/joint_limits.yaml')
           .planning_pipelines(pipelines=['ompl'])
           .to_moveit_configs())

    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bridge, 'launch', 'driver.launch.py')),
        launch_arguments={
            'connection': lc('connection'),
            'ip': lc('ip'),
            'serial_port': lc('serial_port'),
            'calibration_file': calib,
            'rviz': 'false',
        }.items())]

    actions.append(Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[cfg.to_dict(), {'publish_robot_description_semantic': True,
                                    'monitor_dynamics': False}]))

    if lc('rviz').lower() in ('true', '1'):
        actions.append(Node(
            package='rviz2', executable='rviz2', output='log',
            arguments=['-d', os.path.join(share, 'config', 'moveit.rviz')],
            parameters=[cfg.robot_description, cfg.robot_description_semantic,
                        cfg.robot_description_kinematics, cfg.planning_pipelines,
                        cfg.joint_limits]))
    return actions


def generate_launch_description():
    args = {
        'connection': 'fake',          # fake | socket | serial
        'ip': '192.168.0.10',
        'serial_port': '/dev/ttyUSB0',
        'calibration_file': '',        # hand_eye.yaml from mycobot_calibration ('' = nominal)
        'rviz': 'true',
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [OpaqueFunction(function=setup)])
