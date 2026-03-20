import os
from glob import glob

from setuptools import setup


package_name = 'mpc_controller'


setup(
    name=package_name,
    version='0.0.0',
    packages=[
        package_name,
        f'{package_name}.mpc_core',
        f'{package_name}.mpc_solvers',
        f'{package_name}.ros2_nodes',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fengwei',
    maintainer_email='huangfengwei56@gmail.com',
    description='Minimal-stage MPC controller package for ROS2 F1TENTH integration.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mpc_controller_node = mpc_controller.ros2_nodes.mpc_controller_node:main',
        ],
    },
)
