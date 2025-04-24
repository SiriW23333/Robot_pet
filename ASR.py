import rclpy  # ROS 2 Python客户端库
from rclpy.node import Node  # ROS 2节点的基类
from std_srvs.srv import SetBool  # 标准服务类型，用于布尔值请求
from std_msgs.msg import String  # 导入String消息类型
import hashlib  # 哈希算法库
import hmac  # 用于消息认证的密钥哈希库
import base64  # 用于Base64编码/解码的库
from socket import *  # 套接字库，用于网络通信
import json, time, threading  # JSON处理、时间操作和线程库
from websocket import create_connection  # 用于创建WebSocket连接的库
import websocket  # WebSocket库
from urllib.parse import quote  # 用于URL编码的库
import logging  # 日志记录库
import pyaudio  # 音频处理库
import os  # 操作系统相关库
from hand_reader.srv import ASRcmd

class Client():
    def __init__(self):
        # 初始化WebSocket连接的基础URL
        base_url = "ws://rtasr.xfyun.cn/v1/ws"
        ts = str(int(time.time()))  # 获取当前时间戳
        tt = (app_id + ts).encode('utf-8')  # 将app_id和时间戳拼接并编码
        md5 = hashlib.md5()  # 创建MD5哈希对象
        md5.update(tt)  # 更新哈希对象
        baseString = md5.hexdigest()  # 获取哈希值
        baseString = bytes(baseString, encoding='utf-8')  # 转换为字节

        apiKey = api_key.encode('utf-8')  # 编码API密钥
        signa = hmac.new(apiKey, baseString, hashlib.sha1).digest()  # 生成HMAC签名
        signa = base64.b64encode(signa)  # 对签名进行Base64编码
        signa = str(signa, 'utf-8')  # 转换为字符串
        self.end_tag = "{\"end\": true}"  # 定义结束标志

        # 创建WebSocket连接
        self.ws = create_connection(base_url + "?appid=" + app_id + "&ts=" + ts + "&signa=" + quote(signa))
        self.trecv = threading.Thread(target=self.recv)  # 创建接收线程
        self.trecv.start()  # 启动接收线程

        # 音频录制参数
        self.CHUNK = 1280  # 每次读取的音频块大小
        self.FORMAT = pyaudio.paInt16  # 音频格式
        self.CHANNELS = 1  # 音频通道数
        self.RATE = 16000  # 采样率

        self.p = pyaudio.PyAudio()  # 初始化PyAudio对象
        self.talk_input = None  # 初始化音频输入对象

        # 打印所有可用设备（调试用）
        device_count = self.p.get_device_count()  # 获取设备数量
        for i in range(device_count):
            info = self.p.get_device_info_by_index(i)  # 获取设备信息
            print(f" 设备{i}: {info['name']} (: {info['maxInputChannels']})")  # 打印设备信息

        # 输入源
        self.input_device_index = None  # 初始化输入设备索引

        # 寻找ES7210作为输入源
        for i in range(device_count):
            info = self.p.get_device_info_by_index(i)  # 获取设备信息

            # 寻找duplex-audio-i2s1的device0
            if "duplex-audio-i2s1" in info['name'] and info['maxInputChannels'] > 0:
                self.input_device_index = i  # 设置输入设备索引
                break

        if self.input_device_index is None:  # 如果未找到设备
            print("未找到ES7210")  # 打印错误信息
            exit(1)  # 退出程序

        # 打开音频流
        self.stream = self.p.open(format=self.FORMAT,
                                channels=self.CHANNELS,
                                rate=self.RATE,
                                input=True,
                                frames_per_buffer=self.CHUNK,
                                input_device_index=self.input_device_index)

        print("ES7210找到，正在录音中...")  # 打印录音信息
        self.recording = False  # 初始化录音状态

    def send(self):
        try:
            while self.recording:  # 如果正在录音
                data = self.stream.read(self.CHUNK)  # 读取音频数据
                self.ws.send(data)  # 发送音频数据
                time.sleep(0.04)  # 等待一段时间
        finally:
            self.stream.stop_stream()  # 停止音频流
            self.stream.close()  # 关闭音频流
            self.p.terminate()  # 终止PyAudio对象

        self.ws.send(bytes(self.end_tag.encode('utf-8')))  # 发送结束标志
        print("send end tag success")  # 打印发送成功信息

    def recv(self):
        global full_sentence  # 定义全局变量用于存储完整识别结果
        full_sentence = ""  # 初始化完整识别结果

        try:
            while self.ws.connected:  # 如果WebSocket连接仍然有效
                result = str(self.ws.recv())  # 接收数据
                if len(result) == 0:  # 如果接收到的数据为空
                    print("receive result end")  # 打印接收结束信息
                    break

                result_dict = json.loads(result)  # 将接收到的数据解析为字典

                # 握手成功消息
                if result_dict["action"] == "started":
                    print("handshake success, result: " + result)  # 打印握手成功信息

                # 正常识别结果处理
                elif result_dict["action"] == "result":
                    try:
                        data_json = json.loads(result_dict["data"])  # 解析识别结果数据
                        words = []  # 初始化单词列表
                        for rt_block in data_json.get("cn", {}).get("st", {}).get("rt", []):  # 遍历识别结果块
                            for ws in rt_block.get("ws", []):  # 遍历词块
                                if ws.get("cw"):  # 如果存在候选词
                                    words.append(ws["cw"][0]["w"])  # 添加候选词到列表
                        final_text = ''.join(words)  # 拼接识别到的文字

                        # 累加识别到的内容
                        full_sentence += final_text  # 更新完整识别结果
                        print("识别结果:", final_text)  # 打印识别结果
                      
                    except Exception as e:  # 捕获异常
                        print("解析识别结果出错:", e)  # 打印错误信息

                # 错误处理
                elif result_dict["action"] == "error":
                    print("rtasr error: " + result)  # 打印错误信息
                    self.ws.close()  # 关闭WebSocket连接
                    return

        except websocket.WebSocketConnectionClosedException:  # 捕获WebSocket连接关闭异常
            print("receive result end")  # 打印接收结束信息
            
        finally:
            self.talk_input = full_sentence # 完整识别结果
            return 

    def close(self):
        self.ws.close()  # 关闭WebSocket连接
        print("connection closed")  # 打印连接关闭信息

    def start_recording(self):
        self.recording = True  # 设置录音状态为True
        self.thread_send = threading.Thread(target=self.send)  # 创建发送线程,本线程用于接受网页输出
        self.thread_send.start()  # 启动发送线程

    def stop_recording(self):
        self.recording = False  # 设置录音状态为False
        self.stream.stop_stream()  # 停止音频流
        self.stream.close()  # 关闭音频流
        self.p.terminate()  # 终止PyAudio对象
        self.ws.send(bytes(self.end_tag.encode('utf-8')))  # 发送结束标志
        print("send end tag success")  # 打印发送成功信息


class ASRService(Node):
    def __init__(self, client):
        super().__init__('asr_service')  # 初始化ROS 2节点
        self.client_ = client 
        self.srv = self.create_service(ASRcmd, '/talk_input', self.handle_request)  # 创建服务
        self.asr_start = False  # 初始化asr_start状态

    def handle_request(self, request, response):
        if request.data:  # 如果请求数据为True
            if self.asr_start:  # 如果已经在录音中，忽略新的开始录音请求
                self.get_logger().info('Recording is already in progress. Ignoring start request.')  # 打印日志信息
                response.success = False  # 设置响应失败
                response.message = "Recording is already in progress."  # 设置响应消息
            else:  # 如果未在录音中，开始录音
                self.get_logger().info('Received request to start recording.')  # 打印日志信息
                self.asr_start = True  # 设置录音状态为True
                self.client.start_recording()  # 调用客户端的开始录音方法
                response.success = True  # 设置响应成功
                response.message = "Recording started."  # 设置响应消息
        else:
            if not self.asr_start:  # 如果未在录音中，忽略停止录音请求
                self.get_logger().info('No recording in progress. Ignoring stop request.')  # 打印日志信息
                response.success = False  # 设置响应失败
                response.message = "No recording in progress."  # 设置响应消息
            else:  # 如果在录音中，停止录音
                self.get_logger().info('Received request to stop recording.')  # 打印日志信息
                self.asr_start = False  # 设置录音状态为False
                self.client.stop_recording()  # 调用客户端的停止录音方法
                response.success = True  # 设置响应成功
                response.message = "Recording stopped."  # 设置响应消息
        return response  # 返回响应

class Sentence(Node):  
    def __init__(self, client):  
        super().__init__('sentence_topic')  # 初始化ROS 2节点
        self.client = client  # 保存客户端对象
        self.sentence_publisher = self.create_publisher(String, 'sentence', 10)  # 创建发布者
        
    def publish_sentence(self):  # 发布句子的方法
        msg = String()
        
        # 检查 talk_input 是否为空
        if not self.client.talk_input:
            msg.data = "请输出这句话：抱歉，我没听到你说话哦"  # 如果没接收到声音，设置默认消息

        # 发布消息
        self.sentence_publisher.publish(msg)
        self.get_logger().info("Publishing sentence: " + msg.data)


if __name__ == '__main__':
    logging.basicConfig()  # 初始化日志配置

    app_id = "88ac66e0"  # 应用ID
    api_key = "45d8a1cf462e36d29258c788a2d2f6ae"  # API密钥

    client = Client()  

    rclpy.init()  # 初始化ROS 2
    asr_service = ASRService(client)  # 创建ASR服务对象
    rclpy.spin(asr_service)  # 阻塞运行ROS 2节点
    asr_service.destroy_node()  # 销毁ROS 2节点
    
    sentence_publisher = Sentence(client)  # 创建句子发布者对象
    sentence_publisher.publish_sentence()


    rclpy.shutdown()  # 关闭ROS 2
