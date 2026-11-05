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
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smoh',
    maintainer_email='dhtjdalsqkqh@gmail.com',
    description='myCobot 320 driver (socket / serial / fake) for MoveIt2',
    license='BSD',
    entry_points={
        'console_scripts': [
            'mycobot_driver = mycobot_bridge.driver_node:main',
        ],
    },
)
