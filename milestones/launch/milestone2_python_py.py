from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    sim = False  # Set to False when using a real car

    return LaunchDescription([
        Node(
            package='safety_python',
            executable='vision_safety_node_python',
            name='vision_safety_node_launch',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'sim': sim},
                {'ttc_full': 0.60},
                {'distance_full': 0.30},
                {'ttc_partial': 1.0},
                {'auto_release': True}],
        ),

        Node(
            package='gap_following',
            executable='gap_following_node',
            name='gap_following_node_launch',
            output='screen',
            parameters=[{'sim': sim}],
        )
    ])

