from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration('params_file')
    odom_topic = LaunchConfiguration('odom_topic')
    scan_topic = LaunchConfiguration('scan_topic')
    drive_topic = LaunchConfiguration('drive_topic')
    final_drive_topic = LaunchConfiguration('final_drive_topic')
    drive_control_topic = LaunchConfiguration('drive_control_topic')
    publish_drive_control = LaunchConfiguration('publish_drive_control')
    enable_safety = LaunchConfiguration('enable_safety')
    safety_ttc_threshold = LaunchConfiguration('safety_ttc_threshold')
    safety_distance_threshold = LaunchConfiguration('safety_distance_threshold')
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
            'scan_topic',
            default_value='/scan',
            description='LaserScan topic used by the safety node.',
        ),
        DeclareLaunchArgument(
            'drive_topic',
            default_value='/mpc/drive',
            description='Raw Ackermann output topic remap for the MPC controller.',
        ),
        DeclareLaunchArgument(
            'final_drive_topic',
            default_value='/drive',
            description='Final Ackermann drive topic published by the safety node.',
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
            'enable_safety',
            default_value='true',
            description='Whether to launch the safety node alongside the MPC node.',
        ),
        DeclareLaunchArgument(
            'safety_ttc_threshold',
            default_value='0.57',
            description='Time-to-collision threshold for AEB.',
        ),
        DeclareLaunchArgument(
            'safety_distance_threshold',
            default_value='0.25',
            description='Minimum obstacle distance threshold for AEB.',
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
        Node(
            package='safety',
            executable='safety_node',
            name='safety_node',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(enable_safety),
            parameters=[
                {
                    'use_sim_time': use_sim_time,
                    'ttc_threshold': safety_ttc_threshold,
                    'distance_threshold': safety_distance_threshold,
                },
            ],
            remappings=[
                ('/ego_racecar/odom', odom_topic),
                ('/scan', scan_topic),
                ('/drive_control', drive_control_topic),
                ('/drive', final_drive_topic),
            ],
        ),
    ])
