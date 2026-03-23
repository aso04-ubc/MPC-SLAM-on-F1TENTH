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
                {'ttc_full': 0.60}, 
                {'distance_full': 0.30},
                {'partial_brake_decel' : 0.3},
                {'ttc_partial' : 1.0},
                {'aeb_auto_release': True}],
        ),

        Node(
            package='gap_following',
            executable='gap_following_node',
            name='gap_following_node_launch',
            output='screen',
            parameters=[{'sim': sim}],
        )
    ])
