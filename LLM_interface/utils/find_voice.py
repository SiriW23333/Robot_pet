import pyaudio

p = pyaudio.PyAudio()  # 初始化PyAudio对象
talk_input = None  # 初始化音频输入对象

# 打印所有可用设备（调试用）
device_count = p.get_device_count()  # 获取设备数量
for i in range(device_count):
  info = p.get_device_info_by_index(i)  # 获取设备信息
  print(f" 设备{i}: {info['name']} (: {info['maxInputChannels']})")  # 打印设备信息

# 输入源
input_device_index = None  # 初始化输入设备索引

# 寻找ES7210作为输入源
for i in range(device_count):
  info = p.get_device_info_by_index(i)  # 获取设备信息

  # 寻找duplex-audio-i2s1的device0
  if "duplex-audio-i2s1" in info['name'] and info['maxInputChannels'] > 0:
    input_device_index = i  # 设置输入设备索引
    break

  if input_device_index is None:  # 如果未找到设备
    print("未找到ES7210")  # 打印错误信息
    exit(1)  # 退出程序
                                                                                                                                                                                                        
