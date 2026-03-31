from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    sim = True  # Set to False when using a real car

    return LaunchDescription([
        DeclareLaunchArgument(
            'target_laps',
            default_value='3',
            description='Number of laps before issuing stop command',
        ),
        Node(
            package='safety',
            executable='safety_node',
            name='safety_node_launch',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'sim': sim},
                {'ttc_full': 0.45}, 
                {'distance_full': 0.52},
                {'partial_brake_decel' : 0.32},
                {'ttc_partial' : 1.8},
                {'aeb_auto_release': True}],
        ),
        
        Node(
            package='gap_following',
            executable='gap_following_node',
            name='gap_following_node_launch',
            output='screen',
            parameters=[{'sim': sim}],
        ),

        Node(
            package='high_level_planner',
            executable='high_level_planner_node',
            name='high_level_planner_node_launch',
            output='screen',
            parameters=[{'sim': sim}],
        ),

        # Node(
        #     package='lap_counter',
        #     executable='lap_counter_node',
        #     name='lap_counter_node_launch',
        #     output='screen',
        #     parameters=[
        #         {'target_laps': LaunchConfiguration('target_laps')},
        #         {'image_topic': '/camera/color/image_raw'},
        #         {'lap_count_topic': '/lap_count'},
        #         {'lap_time_topic': '/lap_time'},
        #     ],
        # )
    ])
