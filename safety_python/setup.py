from setuptools import setup

package_name = 'safety_python'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yiran',
    maintainer_email='yiranmushroom@gmail.com',
    description='Python safety node with AEB and priority-based command arbitration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'safety_python_node = safety_python.safety_python_node:main',
        ],
    },
)
