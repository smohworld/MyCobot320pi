"""MoveIt2 on the real myCobot 320 Pi: pymycobot driver + move_group + RViz.

Unlike demo.launch.py this starts no ros2_control_node (mock hardware), so the
driver's /joint_states and arm/gripper action servers are the only ones running.

    ros2 launch mycobot_320pi_moveit real_robot.launch.py
    ros2 launch mycobot_320pi_moveit real_robot.launch.py use_fake_robot:=true
    ros2 launch mycobot_320pi_moveit real_robot.launch.py connection:=serial port:=/dev/ttyAMA0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import (generate_move_group_launch,
                                           generate_moveit_rviz_launch,
                                           generate_rsp_launch,
                                           generate_static_virtual_joint_tfs_launch)


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("firefighter", package_name="mycobot_320pi_moveit").to_moveit_configs()

    args = [
        DeclareLaunchArgument('use_fake_robot', default_value='false'),
        DeclareLaunchArgument('connection', default_value='socket', description='serial | socket'),
        DeclareLaunchArgument('port', default_value='/dev/ttyAMA0'),
        DeclareLaunchArgument('host', default_value='10.32.16.122'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
    ]

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('mycobot_320_driver'), 'launch', 'driver.launch.py')),
        launch_arguments={
            'use_fake_robot': LaunchConfiguration('use_fake_robot'),
            'connection': LaunchConfiguration('connection'),
            'port': LaunchConfiguration('port'),
            'host': LaunchConfiguration('host'),
        }.items(),
    )

    return LaunchDescription(args + [
        driver,
        generate_static_virtual_joint_tfs_launch(moveit_config),
        generate_rsp_launch(moveit_config),
        generate_move_group_launch(moveit_config),
        GroupAction([generate_moveit_rviz_launch(moveit_config)],
                    condition=IfCondition(LaunchConfiguration('use_rviz'))),
    ])

