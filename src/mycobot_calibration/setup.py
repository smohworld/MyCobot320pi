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
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smoh',
    maintainer_email='dhtjdalsqkqh@gmail.com',
    description='Automatic eye-in-hand calibration (OpenCV ChArUco) driven by MoveIt2',
    license='BSD',
    entry_points={
        'console_scripts': [
            'handeye = mycobot_calibration.handeye_node:main',
            'camera = mycobot_calibration.camera_node:main',
            'sim_camera = mycobot_calibration.sim_camera_node:main',
            'intrinsics_calib = mycobot_calibration.intrinsics_calib:main',
            'goto_lookdown = mycobot_calibration.goto_lookdown:main',
        ],
    },
)
