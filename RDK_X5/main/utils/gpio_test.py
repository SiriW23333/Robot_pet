import sys
import signal
import time
import serial
import serial.tools.list_ports
import os
import Hobot.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

# Ctrl+C 退出信号处理
def signal_handler(signal, frame):
    print("\n用户中断，程序退出。")
    sys.exit(0)


def serial_test():
    selected_port="/dev/ttyS1"


    # 检查串口是否存在
    try:
        if not os.path.exists(selected_port):
            print(f"串口设备 {selected_port} 不存在！")
            return -1
    except Exception as e:
        print(f"检查串口失败: {e}")
        return -1


    # 输入波特率
    baudrate =9600 

   
    
    ser = serial.Serial(selected_port, baudrate, timeout=1)
    print(f"已打开串口: {selected_port} 波特率: {baudrate}")


    print("开始测试，按 CTRL+C 退出...")
    while 1:
      try:
              
              
              
              test_data = 0x30
              write_num = ser.write(bytes([0x30]))
              print("Send: ", test_data)
  
              if ser.in_waiting > 0:
                  response = ser.read(ser.in_waiting)
                  print(f"收到响应: {response}")
                  
  
              time.sleep(1)  # 控制发送频率
      except KeyboardInterrupt:
          signal_handler(None, None)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    result = serial_test()
    if result != 0:
        print("串口测试失败！")
    else:
        print("串口测试成功！")
