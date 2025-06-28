import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
import subprocess

class PerceptionMonitor(Node):

    def __init__(self):
        super().__init__('hand_reader')
        self.gesture_value = None
        # 配置QoS策略
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        self.qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        try:
            self.process = subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && source /root/Robot_pet/install/setup.bash && /root/Robot_pet/hand_ws/start.sh'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
            )
            if  self.process.poll() is None:
                self.get_logger().info("start.sh 脚本已成功启动。")
            else:
                self.get_logger().error("start.sh 脚本启动失败。")
        except Exception as e:
            self.get_logger().error(f"启动 start.sh 脚本时出错: {e}")
        # 初始化订阅者
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_hand_gesture_detection',  # 根据实际话题修改
            self.listener_callback,
            self.qos_profile)
    
        self.get_logger().info("节点初始化完成，等待数据...")



    def listener_callback(self, msg):
        self.get_logger().info(f"收到消息，共有 {len(msg.targets)} 个target")
        for i, target in enumerate(msg.targets):
            self.get_logger().info(f"Target[{i}]: {target}")
            for j, attribute in enumerate(target.attributes):
                self.get_logger().info(f"  Attribute[{j}]: type={attribute.type}, value={attribute.value}")

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
