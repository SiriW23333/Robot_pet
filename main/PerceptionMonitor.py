import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from threading import Thread, Lock, Condition, Event
import serial, subprocess, sys, os, time
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../face_ws/face_recognition')))
from face_sqlite import get_favorability, set_favorability
from inference import face_recognization
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MQTT')))
from tuya_mqtt import init_mqtt, report_fav, set_fav_callback
import threading
import sys
import select

class PerceptionMonitor(Node):
    def __init__(self, id):
        super().__init__('hand_reader')
        self.lock = Lock()
        self.condition = Condition(self.lock)
        self.waiting_for_response = False
        self.id = id
        self.fav = get_favorability(id)
        self.get_logger().info(f"正在为用户 {id} 初始化节点...")
        self.qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.llm_enabled = False
        self.llm_pid = None
        
        init_mqtt(
            product_key="pgvauagmoqlb5ymz",
            device_id="26785d15b5c0e333abuihd",
            device_secret="1novSR8RExbZDWve",
            ca_certs="/root/Robot_pet/MQTT/iot-device.pem"
        )
        
        # 设置fav值变化的回调函数
        set_fav_callback(self.on_fav_changed)

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
        self.subscription = self.create_subscription(
            PerceptionTargets, '/hobot_hand_gesture_detection', 
            self.gesture_callback, self.qos_profile)
        self.get_logger().info("已订阅手势识别话题，等待消息...")
        while rclpy.ok() :
            time.sleep(0.1)
        self.get_logger().info("手势识别线程结束")

    def gesture_callback(self, msg):
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
                if cmd == 3 and self.fav >= 10 and self.llm_enabled == False:
                    self.llm_enabled = True
                    self.get_logger().info("检测到cmd=3，好感度已达标，解锁大语言模型对话")
                    # 启动外部进程，并监听键盘输入
                    proc = subprocess.Popen(
                        ['python3', '/root/Robot_pet/LLM_interface/voice_assistant_demo.py']
                    )
                    self.llm_pid = proc.pid
                    self.get_logger().info(f"大语言模型进程已启动，PID={self.llm_pid}")
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
            
            while(self.waiting_for_response):
                time.sleep(0.1)
            
            if cmd == 12 or cmd == 13 or cmd == 2:
                self.ser.write(bytes([0x31]))
                time.sleep(2)
            elif cmd == 11:
                self.ser.write(bytes([0x28]))
                time.sleep(2)
        except Exception as e:
            self.get_logger().error(f"发送串口命令时出错: {str(e)}")
        finally:
            self.get_logger().info("串口命令发送完成，等待下一个手势...")
            

    def _get_command(self, gesture_value):
        cmd_map = {
            5: 0x43, #手掌 打招呼
            12: 0x35,   #左转 等待5秒再发31
            13: 0x36,   #右转 等待5秒再发31
            2: 0x32,  #大拇指 前进 等待五秒再发31
            3: 3, 
            4: 0x34, #嘘 趴下
            11: 0x40, #ok 摇尾巴  5秒之后发28
            14: 0x37 #666 摇摆
        }
        return cmd_map.get(gesture_value)

    def serial_listener(self):
        if not hasattr(self, 'ser') or not self.ser.is_open:
            self.get_logger().error("串口监听线程：串口未初始化或未打开。")
            return
        self.get_logger().info("串口监听线程启动")
        start_time = time.time()
        timeout = 5
        try:
            while self.waiting_for_response and (time.time() - start_time) < timeout:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    try:
                        text = data.decode('utf-8', errors='replace')
                    except Exception as e:
                        text = str(data)
                        self.get_logger().error(f"解码串口数据时出错: {e}")
                    self.get_logger().info(f"收到串口消息: {text}")
                    self.fav += 1
                    self.get_logger().info(f"当前好感度: {self.fav}")
                    report_fav(self.fav)
                    self.get_logger().info(f"好感度已上报: {self.fav}")
                    
                    # with self.condition:
                    #     self.condition.notify_all()
                    #     self.get_logger().info("通知等待线程，串口响应已处理")
                    #     self.waiting_for_response = False
                    break
        except Exception as e:
            self.get_logger().error(f"串口监听出错: {str(e)}")
        finally:
            self.waiting_for_response = False
            self.get_logger().info("串口监听线程结束")

    def on_fav_changed(self, new_fav):
        """当收到云端fav值变化时的回调函数"""
        try:
            old_fav = self.fav
            self.fav = new_fav
            self.get_logger().info(f"收到云端fav值变化: {old_fav} -> {new_fav}")
            
            # 同步到数据库
            try:
                set_favorability(self.id, self.fav)
                self.get_logger().info(f"fav值已同步到数据库: {self.fav}")
            except Exception as e:
                self.get_logger().error(f"同步fav值到数据库时出错: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"处理fav值变化时出错: {str(e)}")

    def destroy_node(self):
        if self.llm_pid is not None:
            try:
                self.get_logger().info(f"正在关闭大语言模型进程，PID={self.llm_pid}")
                os.kill(self.llm_pid, 15)  # 发送SIGTERM
                time.sleep(1)

                try:
                    os.kill(self.llm_pid, 0)
                    self.get_logger().warning("SIGTERM未生效，尝试SIGKILL")
                    os.kill(self.llm_pid, 9)
                except OSError:
                    self.get_logger().info("大语言模型进程已关闭")
            except Exception as e:
                self.get_logger().error(f"关闭大语言模型进程时出错: {e}")

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