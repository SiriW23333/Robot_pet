import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets

class PerceptionMonitor(Node):

    def __init__(self):
        super().__init__('hand_reader')
        
        # 配置QoS策略
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
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
      # 遍历所有目标
        for target in msg.targets:
            # 遍历目标的属性列表
            for attribute in target.attributes:
                # 筛选手势类型属性
                if attribute.type == "gesture":
                    gesture_value = attribute.value
                    self.get_logger().info(
                        f"检测到手势: 类型={gesture_value}",
                    )
                    # 可在此处添加手势映射逻辑（如1=挥手，2=握拳）

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点关闭")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
