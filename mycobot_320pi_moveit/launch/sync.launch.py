"""MoveIt2 demo (mock hardware) + sync_node mirroring /joint_states to the real arm.

Open loop alternative to real_robot.launch.py: MoveIt plans and "executes" on
mock hardware and the arm follows. RViz shows the simulated pose.

    ros2 launch mycobot_320pi_moveit sync.launch.py
    ros2 launch mycobot_320pi_moveit sync.launch.py host:=10.32.16.122
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("firefighter", package_name="mycobot_320pi_moveit").to_moveit_configs()
    config = os.path.join(get_package_share_directory('mycobot_320_driver'), 'config', 'driver.yaml')

    args = [
        DeclareLaunchArgument('use_fake_robot', default_value='false'),
        DeclareLaunchArgument('connection', default_value='socket', description='serial | socket'),
        DeclareLaunchArgument('port', default_value='/dev/ttyAMA0'),
        DeclareLaunchArgument('host', default_value='10.32.16.122'),
    ]
    sync = Node(
        package='mycobot_320_driver',
        executable='sync_node',
        name='mycobot_320_sync',
        output='screen',
        parameters=[
            config,
            {
                'use_fake_robot': ParameterValue(LaunchConfiguration('use_fake_robot'), value_type=bool),
                'connection': LaunchConfiguration('connection'),
                'port': LaunchConfiguration('port'),
                'host': LaunchConfiguration('host'),
            },
        ],
    )
    return LaunchDescription(args + [generate_demo_launch(moveit_config), sync])
