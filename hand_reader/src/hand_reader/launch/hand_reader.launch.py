from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hand_reader',
            executable='hand_reader',
            name='hand_reader',
            parameters=[{'sub_topic': '/hobot_hand_gesture_detection'}],
            output='screen',
            emulate_tty=True
        )
    ])

