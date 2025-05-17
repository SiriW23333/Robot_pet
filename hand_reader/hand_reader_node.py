import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from threading import Thread,Lock,Condition
import serial
import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from threading import Thread, Lock, Condition
import serial

class PerceptionMonitor(Node):
    def __init__(self):
        super().__init__('hand_reader')
        self.gesture_value = None
        self.lock = Lock()  # 修复1：添加锁初始化
        self.condition = Condition(self.lock)  # 修复2：添加条件变量初始化
        self.waiting_for_response = False
        
        # 配置QoS策略
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        self.qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # 初始化订阅者
        self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_hand_gesture_detection',
            self.listener_callback,
            self.qos_profile)
        
        # 初始化串口
        try:
            self.ser = serial.Serial('/dev/ttyS1', 9600, timeout=1)
            self.get_logger().info("串口初始化成功")
        except Exception as e:
            self.get_logger().error(f"串口初始化失败: {str(e)}")
            raise
        
        # 启动串口监听线程
        self.serial_thread = Thread(target=self.serial_listener)  # 修复3：创建线程
        self.serial_thread.daemon = True  # 设置为守护线程
        self.serial_thread.start()
        
        self.get_logger().info("节点初始化完成，等待数据...")

    def listener_callback(self, msg):
        #elf.get_logger().info(f"收到感知数据: {len(msg.targets)}个目标")
        with self.lock:
            for target in msg.targets:
                for attribute in target.attributes:
                    if attribute.type == "gesture":
                        self.gesture_value = attribute.value
                        self.get_logger().info(f"检测到手势: 类型={self.gesture_value}")
                        
                        try:
                            # 发送串口指令
                            cmd = self._get_command()
                            if cmd is not None:
                                self.get_logger().info(f"准备发送指令: 0x{cmd:02X}")
                                self.waiting_for_response = True
                                self.ser.write(bytes([cmd]))
                                self.get_logger().info("指令已发送，等待响应...")
                                
                                # 等待响应（最多等待2秒）
                                if not self.condition.wait(timeout=2.0):
                                    self.get_logger().warning("等待响应超时")
                        except Exception as e:
                            self.get_logger().error(f"处理手势时出错: {str(e)}")
                        finally:
                            self.waiting_for_response = False

    def _get_command(self):
        cmd_map = {
            5: 0x43,   # 打招呼
            12: 0x35,  # 左转
            13: 0x36,  # 右转
            2: 0x37,   # 摇摆
            3: 0x40    # 摇尾巴
        }
        return cmd_map.get(self.gesture_value)

    def serial_listener(self):
        self.get_logger().info("串口监听线程启动")
        while rclpy.ok():
            try:
                # 添加调试：打印可用字节数
                available = self.ser.in_waiting
                if available > 0:
                    data = self.ser.read(available)
                    self.get_logger().info(f"收到串口数据: {data.hex()}")
                    
                    with self.lock:
                        if self.waiting_for_response:
                            self.condition.notify()
                            self.get_logger().info("已通知条件变量")
            except Exception as e:
                self.get_logger().error(f"串口监听出错: {str(e)}")
                break

    def destroy_node(self):
        self.get_logger().info("正在关闭节点...")
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
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
 