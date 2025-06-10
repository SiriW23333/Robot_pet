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
        self.gesture_process = None
        self.get_logger().info(f"正在为用户 {id} 初始化节点...")
        self._internal_destroyed_flag = False
        time.sleep(1)
        self.qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        init_mqtt(
            product_key="pgvauagmoqlb5ymz",
            device_id="26785d15b5c0e333abuihd",
            device_secret="1novSR8RExbZDWve",
            ca_certs="/root/Robot_pet/MQTT/iot-device.pem"
        )
        self._init_serial()
        self._send_serial_command(0x43)
        self._start_threads()
        self.get_logger().info("节点初始化完成，开始手势识别...")

    def _init_serial(self):
        try:
            self.ser = serial.Serial('/dev/ttyS1', 9600, timeout=1)
            self.get_logger().info("串口初始化成功")
        except Exception as e:
            self.get_logger().error(f"串口初始化失败: {str(e)}")

    def _start_threads(self):
        self.get_logger().info("开始启动所有线程...")
        gesture_thread = Thread(target=self.gesture_recognition_thread, daemon=True)
        gesture_thread.start()
        self.get_logger().info("手势识别线程已启动")
        self.get_logger().info("所有线程启动完成")

    def gesture_recognition_thread(self):
        self.get_logger().info("手势识别线程启动")
        time.sleep(5)
        self.subscription = self.create_subscription(
            PerceptionTargets, '/hobot_hand_gesture_detection', 
            self.gesture_callback, self.qos_profile)
        self.get_logger().info("已订阅手势识别话题，等待消息...")
        while rclpy.ok() and not self._internal_destroyed_flag:
            time.sleep(0.1)
        self.get_logger().info("手势识别线程结束")

    def gesture_callback(self, msg):
        if self._internal_destroyed_flag:
            return
        with self.lock:
            for target in msg.targets:
                for attribute in target.attributes:
                    if attribute.type == "gesture":
                        gesture_value = attribute.value
                        self.get_logger().info(f"检测到手势: 类型={gesture_value}")
                        self._handle_gesture(gesture_value)

    def _handle_gesture(self, gesture_value):
        try:
            cmd = self._get_command(gesture_value)
            if cmd is not None:
                if cmd == 3 :
                #and self.fav >= 10:
                    self.get_logger().info("检测到cmd=3，挂起本程序，启动大语言模型交互...")
                    subprocess.run(['python', '/root/Robot_pet/LLM_interface/voice_assistant_demo.py'], check=True)
                    self.get_logger().info("外部程序已退出，恢复本程序")
                else:
                    self._send_serial_command(cmd)
        except Exception as e:
            self.get_logger().error(f"处理手势时出错: {str(e)}")

    def _send_serial_command(self, cmd):
        if not hasattr(self, 'ser') or not self.ser.is_open:
            self.get_logger().error("串口未初始化或未打开，无法发送命令。")
        try:
            self.get_logger().info(f"准备发送指令: 0x{cmd:02X}")
            self.waiting_for_response = True
            serial_thread = Thread(target=self.serial_listener, daemon=True)
            serial_thread.start()
            self.ser.write(bytes([cmd]))
            self.get_logger().info("指令已发送，等待响应...")
            with self.condition:
                if not self.condition.wait(timeout=2.0):
                    self.get_logger().warning("等待响应超时")
        except Exception as e:
            self.get_logger().error(f"发送串口命令时出错: {str(e)}")
        finally:
            self.waiting_for_response = False

    def _get_command(self, gesture_value):
        cmd_map = {
            5: 0x43, 
            12: 0x35,
            13: 0x36, 
            2: 0x37,
            3: 3, 
            4: 0x32, 
            11: 0x48, 
            14: 0x41
        }
        return cmd_map.get(gesture_value)

    def serial_listener(self):
        if not hasattr(self, 'ser') or not self.ser.is_open:
            self.get_logger().error("串口监听线程：串口未初始化或未打开。")
            return
        self.get_logger().info("串口监听线程启动")
        start_time = time.time()
        timeout = 10.0
        try:
            while self.waiting_for_response and (time.time() - start_time) < timeout:
                if self._internal_destroyed_flag:
                    break
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self.get_logger().info(f"收到串口数据: {data}")
                    if any(byte >= 48 and byte <= 57 for byte in data):
                        with self.condition:
                            self.fav += 1
                            self.get_logger().info(f"收到数字响应，当前好感度: {self.fav}")
                            report_fav(self.fav)
                            self.get_logger().info(f"好感度已上报: {self.fav}")
                            self.condition.notify()
                            self.waiting_for_response = False
                            break
                time.sleep(0.1)
        except Exception as e:
            self.get_logger().error(f"串口监听出错: {str(e)}")
        finally:
            self.waiting_for_response = False
            self.get_logger().info("串口监听线程结束")

    def destroy_node(self):
        if self._internal_destroyed_flag:
            return
        self._internal_destroyed_flag = True
        self.get_logger().info("正在关闭节点...")
        if self.gesture_process and self.gesture_process.poll() is None:
            self.get_logger().info("正在关闭手势识别服务...")
            try:
                os.killpg(os.getpgid(self.gesture_process.pid), 15)
                self.gesture_process.wait(timeout=2)
            except ProcessLookupError:
                self.get_logger().warning("手势识别服务进程未找到，可能已关闭。")
            except subprocess.TimeoutExpired:
                self.get_logger().warning("手势识别服务关闭超时 (SIGTERM)，尝试 SIGKILL...")
                try:
                    os.killpg(os.getpgid(self.gesture_process.pid), 9)
                    self.get_logger().info("手势识别服务已强制关闭 (SIGKILL)")
                except Exception as e_kill:
                    self.get_logger().error(f"强制关闭手势识别服务时出错: {e_kill}")
            except Exception as e:
                self.get_logger().error(f"关闭手势识别服务时出错: {e}")
            else:
                self.get_logger().info("手势识别服务已关闭")
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("摄像头已释放")
        cv2.destroyAllWindows()
        try:
            set_favorability(self.id, self.fav)
            self.get_logger().info(f"好感度已保存: {self.fav}")
        except Exception as e:
            self.get_logger().error(f"保存好感度时出错: {str(e)}")
        if hasattr(self, 'ser') and self.ser and self.ser.is_open:
            self.ser.close()
            self.get_logger().info("串口已关闭")
        super().destroy_node()
        self.get_logger().info("节点已成功销毁。")

def main(args=None):
    # client_id = face_recognization()
    # if client_id is None:
    #     print("人脸识别失败或用户退出。应用将关闭。")
    #     return
    client_id = 1
    rclpy.init(args=args)
    node = PerceptionMonitor(id=client_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        if node: node.get_logger().error(f"未知错误导致rclpy.spin退出: {e}")
    finally:
        if node and not node._internal_destroyed_flag:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("应用正在退出。")

if __name__ == '__main__':
    main()
