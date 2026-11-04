"""Hand-eye calibration with RViz.

Simulation (fake arm + synthetic camera, runs automatically):
  ros2 launch mycobot_calibration handeye.launch.py sim:=true

Real robot (start your camera driver separately, publishing image_topic/camera_info_topic):
  ros2 launch mycobot_calibration handeye.launch.py sim:=false ip:=<Pi IP>
  ros2 launch mycobot_calibration handeye.launch.py sim:=false connection:=serial \
      serial_port:=/dev/ttyUSB0
  then:  ros2 service call /handeye_calibrator/plan std_srvs/srv/Trigger   (preview)
         ros2 service call /handeye_calibrator/run  std_srvs/srv/Trigger
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim = lc('sim').lower() in ('true', '1')
    share = get_package_share_directory('mycobot_calibration')
    bridge = get_package_share_directory('mycobot_bridge')

    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bridge, 'launch', 'driver.launch.py')),
        launch_arguments={
            'connection': 'fake' if sim else lc('connection'),
            'ip': lc('ip'),
            'serial_port': lc('serial_port'),
            'calibration_file': lc('calibration_file'),
            'rviz': 'false',
        }.items())]

    board_xyz = [float(v) for v in lc('board_xyz').split(',')]
    board_yaw = float(lc('board_yaw'))
    if sim:
        actions.append(Node(
            package='mycobot_calibration', executable='sim_camera', name='sim_camera',
            output='screen',
            parameters=[{'board_center_xyz': board_xyz, 'board_yaw': board_yaw,
                         'image_topic': lc('image_topic'),
                         'camera_info_topic': lc('camera_info_topic')}]))

    actions.append(Node(
        package='mycobot_calibration', executable='handeye', name='handeye_calibrator',
        output='screen',
        parameters=[{
            'auto_start': sim and lc('auto_start').lower() in ('true', '1'),
            'image_topic': lc('image_topic'),
            'camera_info_topic': lc('camera_info_topic'),
            'poses_file': lc('poses_file'),
            'output_dir': lc('output_dir'),
            'board_guess_center_xyz': board_xyz,
            'board_guess_yaw': board_yaw,
            'planner.n_poses': int(lc('n_poses')),
        }]))
    if lc('rviz').lower() in ('true', '1'):
        actions.append(Node(package='rviz2', executable='rviz2', output='screen',
                            arguments=['-d', os.path.join(share, 'rviz', 'handeye.rviz')]))
    return actions


def generate_launch_description():
    args = {
        'sim': 'true',
        'auto_start': 'true',            # sim only
        'connection': 'socket',
        'ip': '192.168.0.10',
        'serial_port': '/dev/ttyUSB0',
        'calibration_file': '',
        'image_topic': '/camera/image_raw',
        'camera_info_topic': '/camera/camera_info',
        'poses_file': '',
        'output_dir': os.path.expanduser('~/handeye_calib'),
        'board_xyz': '0.0,-0.22,0.0',    # board centre in base [m] (sim truth / real guess)
        'board_yaw': '0.0',
        'n_poses': '18',
        'rviz': 'true',
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [OpaqueFunction(function=setup)])
