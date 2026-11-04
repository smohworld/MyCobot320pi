import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import xacro

PACKAGE_NAME = 'mycobot_description'
DEFAULT_MODEL = 'pro_320_pi_2022_force_gripper.urdf.xacro'


def resolve_model_path(model):
    # Absolute/relative paths are used as-is; bare file names are looked up
    # in the installed urdf/mycobot_320_pi_2022 directory.
    if os.path.isfile(model):
        return os.path.abspath(model)
    share_dir = get_package_share_directory(PACKAGE_NAME)
    candidate = os.path.join(share_dir, 'urdf', 'mycobot_320_pi_2022', model)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f'URDF not found: {model} (also tried {candidate})')


def launch_setup(context, *args, **kwargs):
    model_path = resolve_model_path(LaunchConfiguration('model').perform(context))
    print(f'[display.launch] loading {model_path}')

    # xacro also handles plain URDF files, so .urdf and .xacro both work.
    # calibration_file is only passed when set, so models without that arg still load.
    mappings = {}
    calibration_file = LaunchConfiguration('calibration_file').perform(context)
    if calibration_file:
        mappings['calibration_file'] = os.path.abspath(calibration_file)
    robot_description = xacro.process_file(model_path, mappings=mappings).toxml()

    use_gui = LaunchConfiguration('gui')
    rviz_config = LaunchConfiguration('rviz_config')
    # use_jsp:=false -> no joint_state_publisher at all (another node, e.g. a
    # simulator or the real robot bridge, publishes /joint_states)
    use_jsp = LaunchConfiguration('use_jsp').perform(context).lower() in ('true', '1')

    nodes = [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue(robot_description, value_type=str),
            }],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ]
    if use_jsp:
        nodes += [
            Node(
                package='joint_state_publisher_gui',
                executable='joint_state_publisher_gui',
                condition=IfCondition(use_gui),
            ),
            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                condition=UnlessCondition(use_gui),
            ),
        ]
    return nodes


def generate_launch_description():
    share_dir = get_package_share_directory(PACKAGE_NAME)
    return LaunchDescription([
        DeclareLaunchArgument(
            'model',
            default_value=DEFAULT_MODEL,
            description='URDF/xacro file name in urdf/mycobot_320_pi_2022, or a path to any URDF',
        ),
        DeclareLaunchArgument(
            'calibration_file',
            default_value='',
            description='hand-eye / TCP yaml (empty = config/hand_eye_nominal.yaml)',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Use joint_state_publisher_gui sliders',
        ),
        DeclareLaunchArgument(
            'use_jsp',
            default_value='true',
            description='Start joint_state_publisher(_gui); false when another node publishes /joint_states',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz2',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(share_dir, 'rviz', 'display.rviz'),
            description='RViz2 config file',
        ),
        OpaqueFunction(function=launch_setup),
    ])
