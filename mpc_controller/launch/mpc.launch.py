from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    odom_topic = LaunchConfiguration('odom_topic')
    drive_topic = LaunchConfiguration('drive_topic')
    drive_control_topic = LaunchConfiguration('drive_control_topic')
    publish_drive_control = LaunchConfiguration('publish_drive_control')
    reference_mode = LaunchConfiguration('reference_mode')
    path_csv = LaunchConfiguration('path_csv')
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
            description='Raw Ackermann output topic remap for the MPC controller.',
        ),
        DeclareLaunchArgument(
            'drive_control_topic',
            default_value='/drive_control',
            description='Priority-arbitrated drive command topic.',
        ),
        DeclareLaunchArgument(
            'publish_drive_control',
            default_value='true',
            description='Whether to publish DriveControlMessage to the mux chain.',
        ),
        DeclareLaunchArgument(
            'reference_mode',
            default_value='path_csv',
            description='Reference source: auto, path_csv, or constant_speed.',
        ),
        DeclareLaunchArgument(
            'path_csv',
            default_value=PathJoinSubstitution(
                [FindPackageShare('mpc_controller'), 'config', 'example_centerline.csv']
            ),
            description='CSV file that defines the centerline used by the MPC tracker.',
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
                {
                    'use_sim_time': use_sim_time,
                    'reference_mode': reference_mode,
                    'path_csv': path_csv,
                    'publish_drive_control': publish_drive_control,
                    'drive_control_topic': drive_control_topic,
                },
            ],
            remappings=[
                ('/odom', odom_topic),
                ('/mpc/drive', drive_topic),
            ],
        ),
    ])
