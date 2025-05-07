from launch import LaunchDescription
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # 定义启动参数控制是否运行dataset
    run_dataset = LaunchConfiguration('run_dataset', default='true')

    dataset_node = Node(
        package='face_recognition_ros2',
        executable='face_dataset_node',
        parameters=[{'camera_device': '/dev/video2'}],
        condition=IfCondition(run_dataset)
    )

    training_node = Node(
        package='face_recognition_ros2',
        executable='face_training_node',
        parameters=[{'model_save_path': 'trainer/trainer.yml'}]
    )

    recognition_node = Node(
        package='face_recognition_ros2',
        executable='face_recognition_node',
        parameters=[{'confidence_threshold': 85}]
    )

    # 定义事件处理链
    def conditional_launch(context):
        if context.launch_configurations.get('run_dataset', 'true') == 'true':
            return [
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=dataset_node,
                        on_exit=[training_node, recognition_node]
                    )
                )
            ]
        else:
            return [recognition_node]

    return LaunchDescription([
        dataset_node,
        OpaqueFunction(function=conditional_launch),
        LogInfo(msg='Launch sequence configured')
    ])

