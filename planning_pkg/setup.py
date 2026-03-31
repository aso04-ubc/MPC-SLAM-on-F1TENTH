from setuptools import setup

package_name = 'planning_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='poggers',
    maintainer_email='augustin.so@gmail.com',
    description='Race line planner for map-based MPC',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'race_line_planner = planning_pkg.race_line_planner:main',
        ],
    },
)
