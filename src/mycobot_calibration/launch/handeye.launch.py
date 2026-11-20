"""Hand-eye calibration with MoveIt2: driver + move_group + camera + calibrator (+ RViz).

Simulation (fake arm + synthetic camera, runs automatically with auto_start):
  ros2 launch mycobot_calibration handeye.launch.py sim:=true auto_start:=true

Real robot (USB camera via OpenCV, intrinsics from config/camera.yaml by default):
  ros2 launch mycobot_calibration handeye.launch.py connection:=socket ip:=192.168.0.10 \
      camera_device:=0
  other intrinsics: camera_calibration_file:=<yaml>, or camera_calibration_file:=none to
  take them from camera_info_topic
  ros2 launch mycobot_calibration handeye.launch.py connection:=serial serial_port:=/dev/ttyUSB0 ...
  own camera driver instead: camera:=false image_topic:=... (camera_info_topic or the file)
Planning scene: mycobot_tasks place_scene (desk + floor) is applied at start (scene:=false
to skip; scene_args:="--with-bot --bot 0.02,-0.40" for its options).
Regions: mycobot_tasks regions runs too (RViz shelf / TurtleBot boxes; regions:=false to skip).
then
  ros2 service call /handeye_calibrator/plan std_srvs/srv/Trigger   # preview in RViz
  ros2 service call /handeye_calibrator/run  std_srvs/srv/Trigger   # move + capture + solve
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
import shlex

from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
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
    calib_file = os.path.expanduser(lc('camera_calibration_file'))
    if not calib_file:
        calib_file = os.path.join(share, 'config', 'camera.yaml')
    elif calib_file.lower() == 'none':
        calib_file = ''                  # intrinsics from camera_info_topic
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
            'rviz_config': os.path.join(share, 'config', 'handeye.rviz'),
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
                         'height': int(lc('height')),
                         'camera_calibration_file': calib_file}]))

    if on('scene'):
        # desk / floor (+ TurtleBot / shelf via scene_args) from mycobot_tasks place_scene,
        # once move_group is up; it also removes the 1.5 x 1.5 m 'calibration_table'
        actions.append(TimerAction(period=3.0, actions=[Node(
            package='mycobot_tasks', executable='place_scene', name='place_scene',
            output='screen', arguments=['--wait', '60'] + shlex.split(lc('scene_args')))]))

    if on('regions'):
        # shelf / TurtleBot search regions (RViz Interact: drag the corner spheres to resize,
        # the box itself to move it with the same size), saved to ~/.ros/regions.yaml
        actions.append(Node(
            package='mycobot_tasks', executable='regions', name='regions', output='screen'))

    actions.append(Node(
        package='mycobot_calibration', executable='handeye', name='handeye_calibrator',
        output='screen',
        parameters=[os.path.join(share, 'config', 'handeye.yaml'), {
            'board_file': board_file,
            'camera_calibration_file': '' if sim else calib_file,
            'image_topic': lc('image_topic'),
            'camera_info_topic': lc('camera_info_topic'),
            'board_guess_xyz': board_xyz,
            'board_guess_yaw': float(lc('board_yaw')),
            'n_poses': int(lc('n_poses')),
            'poses_file': os.path.expanduser(lc('poses_file')),
            'output_dir': os.path.expanduser(lc('output_dir')),
            'auto_start': on('auto_start'),
        }]))
    return actions


def generate_launch_description():
    args = {
        'sim': 'false',
        'auto_start': 'false',
        'connection': 'socket',          # socket | serial (ignored with sim:=true)
        'ip': '10.42.0.188',
        'serial_port': '/dev/ttyUSB0',
        'gripper_torque': '-1',          # grip force 0..100, -1 = keep the gripper's stored value
        'calibration_file': '',          # '' = mycobot_description/config/hand_eye.yaml
        'board_file': '',                # '' = installed copy of calibration/charuco_board.yaml
        'camera_calibration_file': '',   # '' = installed config/camera.yaml, none = camera_info
        'camera': 'true',                # start the OpenCV USB camera node
        'camera_device': '2',
        'width': '640',
        'height': '480',
        'image_topic': '/camera/image_raw',
        'camera_info_topic': '/camera/camera_info',
        'board_xyz': '0.22,0.0,-0.007',  # board centre guess in base [m] (desk top = -7 mm)
        'board_yaw': '0.0',
        'n_poses': '20',
        'poses_file': '',                # replay a previous run's planned_poses.yaml
        'output_dir': '~/handeye_calib',
        'rviz': 'true',
        'scene': 'true',                 # apply mycobot_tasks place_scene (desk + floor)
        'regions': 'true',               # mycobot_tasks regions (RViz shelf / TurtleBot boxes)
        'scene_args': '',                # place_scene options, e.g. "--with-bot --bot 0.02,-0.40"
    }
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
        + [OpaqueFunction(function=setup)])
