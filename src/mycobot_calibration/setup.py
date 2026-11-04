import os
from glob import glob

from setuptools import setup

package_name = 'mycobot_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smoh',
    maintainer_email='dhtjdalsqkqh@gmail.com',
    description='Automatic eye-in-hand calibration for myCobot 320 (ChArUco, OpenCV)',
    license='BSD',
    entry_points={
        'console_scripts': [
            'handeye = mycobot_calibration.handeye_node:main',
            'sim_camera = mycobot_calibration.sim_camera_node:main',
            'solve_offline = mycobot_calibration.solve_offline:main',
        ],
    },
)
