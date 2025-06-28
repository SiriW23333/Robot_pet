import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import subprocess

class PerceptionMonitor(Node):

    def __init__(self):
        super().__init__('hand_reader')
        self.gesture_value = None
        # 配置QoS策略
        self.qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
      
        
        # 初始化订阅者
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_hand_gesture_detection',  # 根据实际话题修改
            self.listener_callback,
            self.qos_profile)
    
        self.get_logger().info("节点初始化完成，等待数据...")



    def listener_callback(self, msg):
          for target in msg.targets:
              for attribute in target.attributes:
                  if attribute.type == "gesture":
                      self.gesture_value = attribute.value
                      self.get_logger().info(
                          f"检测到手势: 类型={self.gesture_value}"
                      )

def main(args=None):
    rclpy.init(args=args)

    # 创建节点实例
    monitor = PerceptionMonitor()

    try:
        # 启动事件循环，持续接收消息
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass

    # 销毁节点并关闭ROS2上下文
    monitor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
