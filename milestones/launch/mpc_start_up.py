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
                {'ttc_full': 0.50}, 
                {'distance_full': 0.40},
                {'partial_brake_decel' : 0.3},
                {'ttc_partial' : 3.0},
                {'aeb_auto_release': True}],
        ),

        Node(
            package='mpc_controller',
            executable='mpc_controller_node',
            name='mpc_controller_node_launch',
            output='screen',
            parameters=[],
        ),

    ])
