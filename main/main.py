import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from threading import Thread, Lock, Condition, Event
import serial, subprocess, sys, os, time
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../face_ws/opencv')))
from face_sqlite import get_favorability, set_favorability
from inference import face_recognization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MQTT')))
from tuya_mqtt import init_mqtt, report_fav

class PerceptionMonitor(Node):
    def __init__(self, id):
        super().__init__('hand_reader')
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.waiting_for_response = False
        self.id = id
        self.fav = get_favorability(id)
        self.last_fav = self.fav
        
        # 用于管理手势识别进程
        self.gesture_process = None
        
        self.get_logger().info(f"正在为用户 {id} 初始化节点...")
        
        # 等待一段时间确保face_recognization释放了摄像头
        time.sleep(1)
        
        # 启动手势识别服务
        self._start_gesture_recognition_service()
        
        # 初始化摄像头
        self.get_logger().info("正在初始化摄像头...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("摄像头设备未打开")
            exit(1)
        else:
            self.get_logger().info("成功打开摄像头设备 0")

        # 测试摄像头
        if self.cap and self.cap.isOpened():
            ret, test_frame = self.cap.read()
            if ret:
                self.get_logger().info(f"摄像头测试成功，分辨率: {test_frame.shape}")
            else:
                self.get_logger().warning("摄像头测试失败，无法读取帧")

        init_mqtt(
            product_key="pgvauagmoqlb5ymz",
            device_id="26785d15b5c0e333abuihd",
            device_secret="1novSR8RExbZDWve",
            ca_certs="/root/Robot_pet/MQTT/iot-device.pem"
        )

        # 初始化ROS订阅但不立即启动监听
        self.qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        self._init_serial()
        
        # 启动手势识别线程和其他必要线程
        self._start_threads()
        self.get_logger().info("节点初始化完成，开始手势识别...")

    def _start_gesture_recognition_service(self):
        """启动手势识别服务"""
        self.get_logger().info("正在启动手势识别服务...")
        
        try:
            # 方法1：使用bash直接执行脚本
            cmd = ['bash', '/root/Robot_pet/hand_ws/start.sh']
            
            # 启动手势识别服务作为后台进程
            self.gesture_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            self.get_logger().info(f"手势识别服务已启动，进程ID: {self.gesture_process.pid}")
            
            # 等待服务启动
            time.sleep(3)
            
            # 检查进程是否还在运行
            if self.gesture_process.poll() is None:
                self.get_logger().info("手势识别服务启动成功")
            else:
                # 如果进程已经结束，获取错误信息
                stdout, stderr = self.gesture_process.communicate()
                self.get_logger().error(f"手势识别服务启动失败")
                self.get_logger().error(f"stdout: {stdout.decode()}")
                self.get_logger().error(f"stderr: {stderr.decode()}")
                
        except Exception as e:
            self.get_logger().error(f"启动手势识别服务时出错: {str(e)}")

    def _init_serial(self):
        try:
            self.ser = serial.Serial('/dev/ttyS1', 9600, timeout=1)
            self.get_logger().info("串口初始化成功")
        except Exception as e:
            self.get_logger().error(f"串口初始化失败: {str(e)}")
            raise

    def _start_threads(self):
        self.get_logger().info("开始启动所有线程...")
        
        # 启动好感度报告线程
        fav_thread = Thread(target=self.fav_reporter, daemon=True)
        fav_thread.start()
        self.get_logger().info("好感度报告线程已启动")
        
        # 启动手势识别线程
        gesture_thread = Thread(target=self.gesture_recognition_thread, daemon=True)
        gesture_thread.start()
        self.get_logger().info("手势识别线程已启动")
        
        self.get_logger().info("所有线程启动完成")

    def gesture_recognition_thread(self):
        """手势识别主线程"""
        self.get_logger().info("手势识别线程启动")
        
        # 等待手势识别服务完全启动
        time.sleep(5)
        
        # 创建ROS订阅
        self.subscription = self.create_subscription(
            PerceptionTargets, '/hobot_hand_gesture_detection', 
            self.gesture_callback, self.qos_profile)
        
        self.get_logger().info("已订阅手势识别话题，等待消息...")
        
        # 保持线程运行
        while rclpy.ok():
            time.sleep(0.1)

    def gesture_callback(self, msg):
        """手势识别回调函数"""
        with self.lock:
            for target in msg.targets:
                for attribute in target.attributes:
                    if attribute.type == "gesture":
                        gesture_value = attribute.value
                        self.get_logger().info(f"检测到手势: 类型={gesture_value}")
                        
                        # 处理手势并发送串口命令
                        self._handle_gesture(gesture_value)

    def _handle_gesture(self, gesture_value):
        """处理手势并发送串口命令"""
        try:
            cmd = self._get_command(gesture_value)
            if cmd is not None:
                if cmd == 3 and self.fav >= 10:
                    self.get_logger().info("检测到cmd=3，挂起本程序，启动大语言模型交互...")
                    subprocess.run(['python', '/root/Robot_pet/LLM_interface/voice_assistant_demo.py'], check=True)
                    self.get_logger().info("外部程序已退出，恢复本程序")
                elif cmd == 4:
                    self.get_logger().info("销毁当前节点，回到人脸识别")
                    self.destroy_node()
                else:
                    # 发送串口命令并启动监听线程
                    self._send_serial_command(cmd)
                
        except Exception as e:
            self.get_logger().error(f"处理手势时出错: {str(e)}")

    def _send_serial_command(self, cmd):
        """发送串口命令并启动监听线程"""
        try:
            self.get_logger().info(f"准备发送指令: 0x{cmd:02X}")
            
            # 启动串口监听线程
            self.waiting_for_response = True
            serial_thread = Thread(target=self.serial_listener, daemon=True)
            serial_thread.start()
            
            # 发送命令
            self.ser.write(bytes([cmd]))
            self.get_logger().info("指令已发送，等待响应...")
            
            # 等待响应
            with self.condition:
                if not self.condition.wait(timeout=2.0):
                    self.get_logger().warning("等待响应超时")
                else:
                    self.get_logger().info("收到响应")
                    
        except Exception as e:
            self.get_logger().error(f"发送串口命令时出错: {str(e)}")
        finally:
            self.waiting_for_response = False

    def _get_command(self, gesture_value):
        """根据手势值获取对应命令"""
        cmd_map = {
            5: 0x43,   # 打招呼
            12: 0x35,  # 左转
            13: 0x36,  # 右转
            2: 0x37,   # 摇摆
            3: 3,      # 挂起本程序，启动大语言模型交互
            4: 4,      # 销毁当前节点，回到人脸识别
            11: 0x48,  # 伸懒腰
            14: 0x32   # 趴下
        }
        return cmd_map.get(gesture_value)

    def serial_listener(self):
        """串口监听线程 - 只在发送命令后启动"""
        self.get_logger().info("串口监听线程启动")
        
        start_time = time.time()
        timeout = 3.0  # 3秒超时
        
        while self.waiting_for_response and (time.time() - start_time) < timeout:
            try:
                available = self.ser.in_waiting
                if available > 0:
                    data = self.ser.read(available)
                    self.get_logger().info(f"收到串口数据: {data}")
                    
                    if b'OK' in data:
                        with self.condition:
                            self.fav += 1
                            self.get_logger().info(f"收到OK响应，当前好感度: {self.fav}")
                            self.condition.notify()
                            break
                            
                time.sleep(0.1)  # 短暂休眠避免CPU占用过高
                
            except Exception as e:
                self.get_logger().error(f"串口监听出错: {str(e)}")
                break
        
        self.get_logger().info("串口监听线程结束")

    def fav_reporter(self):
        """好感度报告线程"""
        while rclpy.ok():
            if self.fav != self.last_fav:
                report_fav(self.fav)
                self.last_fav = self.fav
            time.sleep(100)  # 每2秒检测一次

    def destroy_node(self):
        self.get_logger().info("正在关闭节点...")
        
        # 关闭手势识别服务
        if self.gesture_process and self.gesture_process.poll() is None:
            self.get_logger().info("正在关闭手势识别服务...")
            try:
                # 终止整个进程组
                os.killpg(os.getpgid(self.gesture_process.pid), 15)  # SIGTERM
                time.sleep(2)
                
                # 如果还没结束，强制杀死
                if self.gesture_process.poll() is None:
                    os.killpg(os.getpgid(self.gesture_process.pid), 9)  # SIGKILL
                    
                self.get_logger().info("手势识别服务已关闭")
            except Exception as e:
                self.get_logger().error(f"关闭手势识别服务时出错: {e}")
        
        # 关闭摄像头
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # 写入好感度到数据库
        try:
            set_favorability(self.id, self.fav)
            self.get_logger().info(f"好感度已保存: {self.fav}")
        except Exception as e:
            self.get_logger().error(f"保存好感度时出错: {str(e)}")
        
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    # 先进行人脸识别
    client_id = face_recognization()
    if client_id is None:
        print("人脸识别失败或用户退出")
        return
    
    # 初始化ROS2
    rclpy.init(args=args)
    
    # 启动手势检测节点
    node = PerceptionMonitor(id=client_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
