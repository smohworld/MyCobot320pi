import os
from setuptools import setup, __version__ as setuptools_version
from packaging.version import Version
from glob import glob

package_name = 'mycobot_description'

# Checking the setuptools version
use_dash_separated_options = Version(setuptools_version) < Version("58.0.0")


# Dynamically generate setup.cfg content
setup_cfg_content = """
[develop]
{script_option}=$base/lib/{package_name}

[install]
{install_scripts_option}=$base/lib/{package_name}
""".format(
    package_name=package_name,
    script_option='script-dir' if use_dash_separated_options else 'script_dir',
    install_scripts_option='install-scripts' if use_dash_separated_options else 'install_scripts'
)

# Write the contents to setup.cfg
with open("setup.cfg", "w") as f:
    f.write(setup_cfg_content)

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch / rviz
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # calibration / mounting yaml (camera extrinsics, TCP)
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # URDF
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        # URDF + visual (.dae) / collision (.stl) meshes
        (os.path.join('share', package_name, 'urdf', 'mycobot_320_pi_2022'),
            glob('urdf/mycobot_320_pi_2022/*.urdf') + glob('urdf/mycobot_320_pi_2022/*.xacro')
            + glob('urdf/mycobot_320_pi_2022/*.dae')
            + glob('urdf/mycobot_320_pi_2022/*.stl')),
        (os.path.join('share', package_name, 'urdf', 'mycobot_320_pi_2022', 'force_control_gripper'),
            glob('urdf/mycobot_320_pi_2022/force_control_gripper/*.dae')
            + glob('urdf/mycobot_320_pi_2022/force_control_gripper/*.stl'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='u2',
    maintainer_email='u2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
