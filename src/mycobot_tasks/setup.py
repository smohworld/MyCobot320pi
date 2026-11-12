from setuptools import setup

package_name = 'mycobot_tasks'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smoh',
    maintainer_email='dhtjdalsqkqh@gmail.com',
    description='Pick / hover tasks (red cube, tray) on top of the calibrated myCobot 320',
    license='BSD',
    entry_points={
        'console_scripts': [
            'red_cube_hover = mycobot_tasks.red_cube_hover:main',
            'click_move = mycobot_tasks.click_move:main',
            'tray_detect = mycobot_tasks.tray_detect:main',
            'tray_pick = mycobot_tasks.tray_pick:main',
        ],
    },
)
