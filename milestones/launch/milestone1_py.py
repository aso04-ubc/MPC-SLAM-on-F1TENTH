from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='safety_python',
            executable='safety_python_node',
            name='safety_node_launch',
            output='screen',
            emulate_tty=True,
        ),

        Node(
            package='wall_follow',
            executable='wall_follow_node',
            name='wall_follow_node_launch',
            output='screen',
        )
    ])