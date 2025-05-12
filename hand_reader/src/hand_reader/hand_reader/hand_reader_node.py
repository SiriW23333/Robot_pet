import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from threading import Thread,Lock,Condition


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

        #初始化串口
    self.ser1 = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # 根据实际串口修改


    def listener_callback(self, msg):
    with self.lock:
        for target in msg.targets:
            for attribute in target.attributes:
                if attribute.type == "gesture":
                    self.gesture_value = attribute.value
                    self.get_logger().info(
                        f"检测到手势: 类型={self.gesture_value}"
                    )
                    # 检测到特定手势后发送串口指令并等待响应
                    if self.gesture_value is not None:
                        self.waiting_for_response = True
                        self.send_serial_command()
                        self.condition.wait()  # 阻塞直到被唤醒
    
    def send_serial_command(self):
        if self.gesture_value ==5:
            cmd = 0x43
        elif self.gesture_value == 12:
            cmd = 0x35
        elif self.gesture_value == 13:
            cmd = 0x36
        elif self.gesture_value == 2:
            cmd = 0x37
        elif self.gesture_value == 3:
            cmd = 0x40

        if self.ser.is_open:
            self.ser.write(bytes([cmd]))  # 发送二进制指令
            self.get_logger().info("已发送串口指令")

    def serial_listener(self):
        while True:
            data = ser.read(ser.in_waiting or 2).decode('UTF-8')
            if data == b"OK":  # 收到下位机确认
                with self.lock:
                    self.waiting_for_response = False
                    self.condition.notify()  # 唤醒手势线程
                    self.get_logger().info("收到下位机响应，已唤醒手势识别")

    def destroy_node(self):
        self.ser.close()
        self.serial_thread.join()
        super().destroy_node()
    '''
手势	说明	数值
ThumbUp	竖起大拇指	2  摇摆
Victory	V手势	3  
Mute	"嘘"手势	4
Palm	手掌	5  打招呼
Okay	OK手势	11  唤醒蓝牙语音
ThumbLeft	大拇指向左	12  左转
ThumbRight	大拇指向右	13  右转
Awesome	666手势	14 
'''

'''
下位机控制指令
0x33 前进
0x35 左转
0x36 右转
0x37 摇摆
0x40 摇尾巴
0x43 打招呼
'''
 

'''class ASRClient(Node):
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

'''
