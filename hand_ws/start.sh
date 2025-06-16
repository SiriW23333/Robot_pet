#! /bin/bash

# 配置tros.b环境
source /opt/tros/humble/setup.bash
source /opt/ros/humble/setup.bash
# 配置USB摄像头
export CAM_TYPE=usb

# 启动launch文件
ros2 launch hand_gesture_detection hand_gesture_detection.launch.py width:=640 height:=480 
