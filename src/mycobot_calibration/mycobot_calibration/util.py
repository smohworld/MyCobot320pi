import os

from ament_index_python.packages import get_package_share_directory


def default_board_file():
    return os.path.join(get_package_share_directory('mycobot_calibration'),
                        'config', 'charuco_board.yaml')


def default_description_calibration():
    return os.path.join(get_package_share_directory('mycobot_description'),
                        'config', 'hand_eye_nominal.yaml')
