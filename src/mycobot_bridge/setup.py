import os
from glob import glob

from setuptools import setup

package_name = 'mycobot_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'robot_side'), glob('robot_side/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smoh',
    maintainer_email='dhtjdalsqkqh@gmail.com',
    description='ROS 2 driver for myCobot 320 + myGripper F100 using pymycobot',
    license='BSD',
    entry_points={
        'console_scripts': [
            'driver = mycobot_bridge.driver_node:main',
            'move_joints = mycobot_bridge.move_joints:main',
        ],
    },
)
