import paho.mqtt.client as mqtt
import json
import time
from generate_pswd import sign_in

# 涂鸦 MQTT 配置
broker = "m1.tuyacn.com"
port = 8883
client_id = "26785d15b5c0e333abuihd "  # 替换为你的 DeviceId

username,password = sign_in()

# DP点定义（示例：DPID 1 表示开关状态）
dp_switch = {"1": True}  # 控制灯开
dp_data = {"2": 25.5}     # 温度传感器数据

# 创建客户端
client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)

# 设置用户名密码
client.username_pw_set(username, password)

# 连接回调
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(f"$aws/things/{client_id}/shadow/update/accepted")

# 接收消息回调
def on_message(client, userdata, msg):
    print(f"Received message: {msg.topic} {str(msg.payload)}")
    if "$aws/things" in msg.topic:
        try:
            payload = json.loads(msg.payload)
            if "state" in payload:
                print("Command received:", payload["state"])
        except Exception as e:
            print("Parse error:", e)

# 设置回调函数
client.on_connect = on_connect
client.on_message = on_message

# 连接服务器
client.connect(broker, port, keepalive=60)

# 启动网络循环
client.loop_start()

# 主循环：定期上报数据
try:
    while True:
        payload = {
            "reported": {
                "time": int(time.time()),
                **dp_switch,
                **dp_data
            }
        }
        client.publish(f"$aws/things/{client_id}/shadow/update", json.dumps(payload))
        print("Published data to Tuya Cloud")
        time.sleep(10)
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()