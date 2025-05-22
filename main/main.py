import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from threading import Thread, Lock, Condition
import serial, subprocess, sys, os, time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../face_ws/client_sqlite')))
from face_sqlite import get_favorability, set_favorability
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MQTT')))
from tuya_mqtt import init_mqtt, report_fav

class PerceptionMonitor(Node):
    def __init__(self, name):
        super().__init__('hand_reader')
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.waiting_for_response = False
        self.name = name
        self.fav = get_favorability(name)
        self.last_fav = self.fav

        init_mqtt(
            product_key="pgvauagmoqlb5ymz",
            device_id="26785d15b5c0e333abuihd",
            device_secret="1novSR8RExbZDWve",
            ca_certs="/root/Robot_pet/MQTT/iot-device.pem"
        )

        self.qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscription = self.create_subscription(
            PerceptionTargets, '/hobot_hand_gesture_detection', self.listener_callback, self.qos_profile)

        self._init_serial()
        self._start_threads()
        self.get_logger().info("节点初始化完成，等待数据...")

    def _init_serial(self):
        try:
            self.ser = serial.Serial('/dev/ttyS1', 9600, timeout=1)
            self.get_logger().info("串口初始化成功")
        except Exception as e:
            self.get_logger().error(f"串口初始化失败: {str(e)}")
            raise

    def _start_threads(self):
        Thread(target=self.serial_listener, daemon=True).start()
        Thread(target=self.fav_reporter, daemon=True).start()

    def listener_callback(self, msg):
        with self.lock:
            for target in msg.targets:
                for attribute in target.attributes:
                    if attribute.type == "gesture":
                        self.gesture_value = attribute.value
                        self.get_logger().info(f"检测到手势: 类型={self.gesture_value}")
                        self._handle_gesture()

    def _handle_gesture(self):
        try:
            cmd = self._get_command()
            if cmd is not None:
                if cmd == 3 and self.fav >= 10:
                    self.get_logger().info("检测到cmd=3，挂起本程序，启动大语言模型交互...")
                    subprocess.run(['python', '/root/Robot_pet/LLM_interface/voice_assistant_demo.py'], check=True)
                    self.get_logger().info("外部程序已退出，恢复本程序")
                elif cmd == 4:
                    ##TODO:销毁当前节点，回到人脸识别
                    self.get_logger().info("销毁当前节点，回到人脸识别")
                    self.destroy_node()
                else :
                    self.get_logger().info(f"准备发送指令: 0x{cmd:02X}")
                    self.waiting_for_response = True
                    self.ser.write(bytes([cmd]))
                    self.get_logger().info("指令已发送，等待响应...")
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
            3: 3,      #挂起本程序，启动大语言模型交互
            4: 4,      #销毁当前节点，回到人脸识别
            11: 0x48, # 伸懒腰
            14: 0x32   # 趴下
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
                    self.get_lo gger().info(f"收到串口数据: {data}")
                    if b'OK' in data:
                        with self.lock:
                            self.fav += 1
                            self.get_logger().info(f"收到OK响应，当前好感度: {self.fav}")
                            if self.waiting_for_response:
                                self.condition.notify()
                                self.get_logger().info("已通知条件变量")

            except Exception as e:
                self.get_logger().error(f"串口监听出错: {str(e)}")
                break

    def fav_reporter(self):
        while True:
            if self.fav != self.last_fav:
                report_fav(self.fav)
                self.last_fav = self.fav
            time.sleep(2)  # 每2秒检测一次

    def destroy_node(self):
        self.get_logger().info("正在关闭节点...")
        # 写入好感度到数据库
        try:
            set_favorability(self.name,self.fav)
            self.get_logger().info(f"好感度已保存: {self.fav}")
        except Exception as e:
            self.get_logger().error(f"保存好感度时出错: {str(e)}")
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    ##TODO: 这里需要由人脸识别的处理
    rclpy.init(args=args)
    node = PerceptionMonitor(name="your_name")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
