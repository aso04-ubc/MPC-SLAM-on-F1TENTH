from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim = LaunchConfiguration('sim')
    use_race_line_planner = LaunchConfiguration('use_race_line_planner')
    odom_topic = LaunchConfiguration('odom_topic')
    imu_topic = LaunchConfiguration('imu_topic')
    scan_topic = LaunchConfiguration('scan_topic')

    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='true'),
        DeclareLaunchArgument('use_race_line_planner', default_value='true'),
        DeclareLaunchArgument('odom_topic', default_value='/odom'),
        DeclareLaunchArgument('imu_topic', default_value='/sensors/imu/raw'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),

        Node(
            package='perception_pkg',
            executable='live_mapper_node',
            name='live_mapper_node_launch',
            output='screen',
            parameters=[
                {'sim': sim},
                {'odom_topic': odom_topic},
                {'scan_topic': scan_topic},
                {'imu_topic': imu_topic},
                {'map_topic': '/mapping/occupancy_grid'},
                {'pose_topic': '/mapping/fused_pose'},
                {'map_publish_rate_hz': 2.0},
                {'pose_publish_rate_hz': 2.0},
                {'icp_enabled': False},
                {'show_opencv_debug': True},
            ],
        ),

        Node(
            package='safety',
            executable='safety_node',
            name='safety_node_launch',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'sim': sim},
                {'ttc_threshold': 0.60},
                {'distance_threshold': 0.30},
            ],
        ),

        Node(
            package='planning_pkg',
            executable='race_line_planner',
            name='race_line_planner_launch',
            output='screen',
            parameters=[
                {'sim': sim},
                {'map_topic': '/mapping/occupancy_grid'},
                {'pose_topic': '/mapping/fused_pose'},
            ],
        ),

        Node(
            package='mpc_controller',
            executable='mpc_controller_node',
            name='mpc_controller_launch',
            output='screen',
            parameters=[
                {'sim': sim},
                {'use_race_line_planner': use_race_line_planner},
                {'odom_topic': odom_topic},
            ],
        ),
    ])
