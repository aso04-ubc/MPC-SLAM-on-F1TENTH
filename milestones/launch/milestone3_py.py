from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    sim = False  # Set to False when using a real car

    return LaunchDescription([
        Node(
            package='safety',
            executable='safety_node',
            name='safety_node_launch',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'sim': sim},
                {'ttc_full': 0.70}, 
                {'distance_full': 0.40},
                {'partial_brake_decel' : 0.3},
                {'ttc_partial' : 1.0},
                {'aeb_auto_release': True}],
        ),

        Node(
            package='camera_gap_follow',
            executable='camera_gap_follow_node',
            name='camera_gap_follow_node_launch',
            output='screen',
            parameters=[],
        ),

        Node(
            package='lap_counter',
            executable='lap_counter_node',
            name='lap_counter_node_launch',
            output='screen',
            parameters=[
                {'target_laps': 3},
                {'image_topic': '/camera/color/image_raw'},
                {'lap_count_topic': '/lap_count'},
                {'lap_time_topic': '/lap_time'},
            ],
        )
    ])
