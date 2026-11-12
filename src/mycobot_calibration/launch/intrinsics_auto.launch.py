"""Automatic intrinsics calibration: arm moves the flange camera over the board on the desk.

Simulation (fake arm + synthetic camera):
  ros2 launch mycobot_calibration intrinsics_auto.launch.py sim:=true auto_start:=true

Real robot (put the camera over the board first: ros2 run mycobot_calibration goto_lookdown):
  ros2 launch mycobot_calibration intrinsics_auto.launch.py connection:=serial \
      serial_port:=/dev/ttyUSB0 board_xyz:=0.22,0.0,-0.007
  ros2 service call /intrinsics_auto/run std_srvs/srv/Trigger      # the arm moves!
Result: output_dir/<time>/camera.yaml (+ images/, report.txt); copy it to
src/mycobot_calibration/config/camera.yaml yourself.
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
    on = lambda n: lc(n).lower() in ('true', '1')  # noqa: E731
    sim = on('sim')
    share = get_package_share_directory('mycobot_calibration')
    moveit = get_package_share_directory('mycobot_moveit_config')
    board_file = os.path.expanduser(lc('board_file')) or os.path.join(
        share, 'config', 'charuco_board.yaml')
    board_xyz = [float(v) for v in lc('board_xyz').split(',')]

    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit, 'launch', 'bringup.launch.py')),
        launch_arguments={
            'connection': 'fake' if sim else lc('connection'),
            'ip': lc('ip'),
            'serial_port': lc('serial_port'),
            'gripper_torque': lc('gripper_torque'),
            'calibration_file': lc('calibration_file'),
            'rviz': lc('rviz'),
            'rviz_config': os.path.join(share, 'config', 'intrinsics_auto.rviz'),
        }.items())]
    if sim:
        actions.append(Node(
            package='mycobot_calibration', executable='sim_camera', output='screen',
            parameters=[{'board_file': board_file, 'board_center_xyz': board_xyz,
                         'board_yaw': float(lc('board_yaw'))}]))
    elif on('camera'):
        actions.append(Node(
            package='mycobot_calibration', executable='camera', output='screen',
            parameters=[{'device': lc('camera_device'), 'width': int(lc('width')),
                         'height': int(lc('height'))}]))
    actions.append(Node(
        package='mycobot_calibration', executable='intrinsics_auto', output='screen',
        parameters=[{
            'board_file': board_file,
            'prior_intrinsics_file': os.path.expanduser(lc('prior_intrinsics_file')),
            'focal_px_guess': float(lc('focal_px_guess') or (600.0 if sim else 470.0)),
            'image_topic': lc('image_topic'),
            'board_guess_xyz': board_xyz,
            'board_guess_yaw': float(lc('board_yaw')),
            'target_coverage': float(lc('target_coverage')),
            'min_views': int(lc('min_views')),
            'output_dir': os.path.expanduser(lc('output_dir')),
            'auto_start': on('auto_start'),
        }]))
    return actions


def generate_launch_description():
    args = {
        'sim': 'false',
        'auto_start': 'false',
        'connection': 'socket',
        'ip': '192.168.0.10',
        'serial_port': '/dev/ttyUSB0',
        'gripper_torque': '-1',
        'calibration_file': '',          # '' = mycobot_description/config/hand_eye.yaml
        'board_file': '',
        'prior_intrinsics_file': '',     # '' = pinhole f=470 only to predict visibility
        'focal_px_guess': '',            # '' = 470 (real camera), 600 with sim:=true
        'camera': 'true',
        'camera_device': '0',
        'width': '640',
        'height': '480',
        'image_topic': '/camera/image_raw',
        'board_xyz': '0.22,0.0,-0.007',
        'board_yaw': '0.0',
        'target_coverage': '0.85',
        'min_views': '25',
        'output_dir': '~/camera_calib_auto',
        'rviz': 'true',
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [OpaqueFunction(function=setup)])
