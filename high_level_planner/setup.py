from setuptools import setup
from glob import glob
import os

package_name = 'high_level_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhou',
    maintainer_email='guanxu@student.ubc.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'high_level_planner_node = high_level_planner.high_level_planner:main',
        ],
    },
)
