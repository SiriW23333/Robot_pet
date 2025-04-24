import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from hand_reader.srv import ASRcmd

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
                    self.gesture_value = attribute.value
                    self.get_logger().info(
                        f"检测到手势: 类型={gesture_value}",
                    )
                    # 可在此处添加手势映射逻辑（如1=挥手，2=握拳）

class ASRClient(Node):
    def __init__(self, monitor_node):
        super().__init__('asr_client')
        self.monitor_node = monitor_node
        self.cli = self.create_client(ASRCmd, '/talk_input')  # 创建客户端
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('服务未就绪，等待中...')

    def send_request(self):
        req = ASRCmd.Request()
        if self.monitor_node.gesture_value == 5:
            req.command = 1  # 开始录音
        elif self.monitor_node.gesture_value == 11:
            req.command = 2  # 停止录音
        else:
            req.command = 0  # 无操作
        
        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'响应: {response.success}, 消息: {response.message}')
        except Exception as e:
            self.get_logger().error(f'服务调用失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    monitor_node = PerceptionMonitor()
    try:
        rclpy.spin(monitor_node)
    except KeyboardInterrupt:
        monitor_node.get_logger().info("节点关闭")
    finally:
        monitor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
