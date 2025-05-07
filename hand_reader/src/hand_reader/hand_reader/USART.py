import sys
import signal
import os
import time

# 导入python串口库
import serial
import serial.tools.list_ports

def signal_handler(signal, frame):
    sys.exit(0)

def list_serial_ports():
    """
    列出所有可用的串口设备
    """
    ports = serial.tools.list_ports.comports()
    available_ports = [port.device for port in ports]
    print("可用串口设备:")
    for port in available_ports:
        print(port)
    return available_ports

def find_serial_port(port_name):
    """
    查找指定名称的串口设备
    :param port_name: 串口设备名 (如 COM3 或 /dev/ttyUSB0)
    :return: 串口对象或 None
    """
    ports = list_serial_ports()
    if port_name in ports:
        print(f"找到串口: {port_name}")
        return port_name
    else:
        print(f"未找到串口: {port_name}")
        return None


def init_serial(port, baudrate=115200, timeout=1):
    """
    初始化串口
    :param port: 串口设备名 (如 COM3 或 /dev/ttyUSB0)
    :param baudrate: 波特率 (默认 115200)
    :param timeout: 超时时间 (默认 1 秒)
    :return: 串口对象
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"串口 {port} 初始化成功，波特率: {baudrate}")
        return ser
    except Exception as e:
        print(f"串口初始化失败: {e}")
        return None

def send_data(ser, data):
    """
    通过串口发送数据
    :param ser: 串口对象
    :param data: 要发送的数据 (字符串)
    """
    try:
        write_num = ser.write(data.encode('UTF-8'))
        print(f"发送数据: {data}")
        return write_num
    except Exception as e:
        print(f"发送数据失败: {e}")
        return 0

def receive_data(ser, size=1024):
    """
    通过串口接收数据
    :param ser: 串口对象
    :param size: 要接收的数据大小 (默认 1024 字节)
    :return: 接收到的数据 (字符串)
    """
    try:
        received_data = ser.read(size).decode('UTF-8')
        print(f"接收数据: {received_data}")
        return received_data
    except Exception as e:
        print(f"接收数据失败: {e}")
        return ""

def close_serial(ser):
    """
    关闭串口
    :param ser: 串口对象
    """
    try:
        ser.close()
        print("串口已关闭")
    except Exception as e:
        print(f"关闭串口失败: {e}")

def RDK2stm(gesture_value):
    if find_serial_port('/dev/ttyUSB0'):
        ser = init_serial('/dev/ttyUSB0')
        if ser:
            # 发送数据
            send_data(ser, gesture_value.to_bytes(4, byteorder='little', signed=True))
           

def stm2RDK():
    if find_serial_port('/dev/ttyUSB0'):
        ser = init_serial('/dev/ttyUSB0')
        if ser:
            # 接收数据
            data = receive_data(ser, 4)
            if data:
                return int.from_bytes(data, byteorder='little', signed=True)