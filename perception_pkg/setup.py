from setuptools import setup

package_name = 'perception_pkg'

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
    description='Live mapping and fused pose publisher',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'live_mapper_node = perception_pkg.live_mapper_node:main',
        ],
    },
)
