from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    odom_topic = LaunchConfiguration('odom_topic')
    drive_topic = LaunchConfiguration('drive_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution(
                [FindPackageShare('mpc_controller'), 'config', 'mpc_params.yaml']
            ),
            description='Optional override for the MPC parameter file.',
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Odometry topic remap for the MPC controller.',
        ),
        DeclareLaunchArgument(
            'drive_topic',
            default_value='/mpc/drive',
            description='Output drive topic remap for the MPC controller.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if available.',
        ),
        Node(
            package='mpc_controller',
            executable='mpc_controller_node',
            name='mpc_controller_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                params_file,
                {'use_sim_time': use_sim_time},
            ],
            remappings=[
                ('/odom', odom_topic),
                ('/mpc/drive', drive_topic),
            ],
        ),
    ])
