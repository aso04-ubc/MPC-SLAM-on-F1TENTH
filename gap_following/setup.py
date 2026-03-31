"""
Setup script for the gap_following ROS 2 package.

This script configures the gap_following package for installation and distribution.
It defines package metadata, dependencies, and entry points for the ROS 2 build system.
"""

from setuptools import setup

package_name = 'gap_following'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'config.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='poggers',
    maintainer_email='augustin.so@gmail.com',
    description='ROS 2 package for LiDAR-based gap following autonomous navigation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gap_following_node = gap_following.main_node:main'
        ],
    },
)
