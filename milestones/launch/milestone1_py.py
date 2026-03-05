from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    sim = True  # Set to False when using a real car

    return LaunchDescription([
        Node(
            package='safety',
            executable='safety_node',
            name='safety_node_launch',
            output='screen',
            emulate_tty=True,
            parameters=[{'ttc_threshold': 0.57}, {'distance_threshold': 0.25}, {"aeb_auto_release" : True}],
        ),

        Node(
            package='wall_follow',
            executable='wall_follow_node',
            name='wall_follow_node_launch',
            output='screen',
            parameters=[{'sim': sim}],
        )
    ])
