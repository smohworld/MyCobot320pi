"""myCobot 320 driver + robot_state_publisher (+ RViz).

  ros2 launch mycobot_bridge driver.launch.py ip:=192.168.0.10              # socket (default)
  ros2 launch mycobot_bridge driver.launch.py connection:=serial serial_port:=/dev/ttyUSB0
  ros2 launch mycobot_bridge driver.launch.py connection:=fake               # no hardware
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bridge_share = get_package_share_directory('mycobot_bridge')
    desc_share = get_package_share_directory('mycobot_description')
    args = {
        'params_file': os.path.join(bridge_share, 'config', 'driver.yaml'),
        'connection': 'socket',
        'ip': '192.168.0.10',
        'netport': '9000',
        'serial_port': '/dev/ttyUSB0',
        'calibration_file': '',
        'rviz': 'true',
    }
    decl = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]

    driver = Node(
        package='mycobot_bridge',
        executable='driver',
        name='mycobot_driver',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'connection': LaunchConfiguration('connection'),
                'ip': LaunchConfiguration('ip'),
                'netport': LaunchConfiguration('netport'),
                'serial_port': LaunchConfiguration('serial_port'),
            },
        ],
    )

    # robot_state_publisher (+ RViz) from the description package; no joint_state_publisher,
    # the driver publishes /joint_states.
    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, 'launch', 'display.launch.py')),
        launch_arguments={
            'use_jsp': 'false',
            'calibration_file': LaunchConfiguration('calibration_file'),
            'rviz': LaunchConfiguration('rviz'),
        }.items(),
    )
    return LaunchDescription(decl + [driver, description])
