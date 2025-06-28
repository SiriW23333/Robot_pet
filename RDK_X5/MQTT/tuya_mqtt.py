import time
import hmac
import hashlib
import json
import paho.mqtt.client as mqtt
import threading

class TuyaMqttSign:
    def __init__(self):
        self.username = ""
        self.password = ""
        self.client_id = ""

    def calculate(self, product_key, device_id, device_secret):
        if not product_key or not device_id or not device_secret:
            return
        timestamp = str(int(time.time() * 1000))
        self.username = f"{device_id}|signMethod=hmacSha256,timestamp={timestamp},secureMode=1,accessType=1"
        self.client_id = f"tuyalink_{device_id}"
        plain_passwd = f"deviceId={device_id},timestamp={timestamp},secureMode=1,accessType=1"
        self.password = self.hmac_sha256(plain_passwd, device_secret)

    @staticmethod
    def hmac_sha256(plain_text, key):
        digest = hmac.new(key.encode(), plain_text.encode(), hashlib.sha256).hexdigest()
        return digest.zfill(64)

# 全局变量保存MQTT客户端和设备ID
_mqtt_client = None
_device_id = None
# 设备属性存储
_device_properties = {
    "fav": 0,  # 好感度
}

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    topic_reply = f"tylink/{userdata['device_id']}/thing/property/set"
    client.subscribe(topic_reply)
    print("Subscribed to:", topic_reply)

# 存储最新的fav值和事件
_latest_fav = None
_fav_event = threading.Event()
_fav_callback = None

def on_message(client, userdata, msg):
    global _latest_fav, _fav_callback
    print(f"Topic: {msg.topic}, Payload: {msg.payload.decode()}")
    
    # 处理属性设置消息
    if msg.topic.endswith("/thing/property/set"):
        fav_value = handle_property_set(msg.payload.decode())
        if fav_value is not None:
            _latest_fav = fav_value
            print(f"更新最新fav值: {_latest_fav}")
            # 触发事件通知
            _fav_event.set()
            # 如果有回调函数，直接调用
            if _fav_callback:
                _fav_callback(fav_value)

def handle_property_set(payload):
    """处理属性设置命令，返回fav值"""
    try:
        # 解析JSON消息
        message = json.loads(payload)
        data = message.get("data", {})
        
        print(f"收到属性设置命令，data={data}")
        
        # 检查是否有fav属性
        if "fav" in data:
            fav_value = data["fav"]
            print(f"收到fav值: {fav_value}")
            return fav_value
        
        return None
        
    except json.JSONDecodeError as e:
        print(f"解析属性设置消息失败: {e}")
        return None
    except Exception as e:
        print(f"处理属性设置消息时出错: {e}")
        return None

def set_fav_callback(callback_func):
    """设置fav值变化的回调函数"""
    global _fav_callback
    _fav_callback = callback_func
    print(f"已设置fav回调函数: {callback_func.__name__}")

def wait_for_fav_change(timeout=None):
    """等待fav值变化，返回新的fav值"""
    global _latest_fav, _fav_event
    if _fav_event.wait(timeout):
        _fav_event.clear()  # 清除事件状态
        fav_value = _latest_fav
        _latest_fav = None  # 获取后清除
        print(f"等待到fav值变化: {fav_value}")
        return fav_value
    return None

def init_mqtt(product_key, device_id, device_secret, ca_certs="./iot-device.pem"):
    global _mqtt_client, _device_id
    _device_id = device_id
    sign = TuyaMqttSign()
    sign.calculate(product_key, device_id, device_secret)
    client = mqtt.Client(client_id=sign.client_id, userdata={"device_id": device_id})
    client.username_pw_set(sign.username, sign.password)
    client.tls_set(ca_certs=ca_certs)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("m1.tuyacn.com", 8883, 60)
    client.loop_start()
    _mqtt_client = client
    print("Tuya MQTT已初始化")

def report_fav(fav):
    global _mqtt_client, _device_id, _device_properties
    if _mqtt_client is None or _device_id is None:
        print("MQTT未初始化，无法上报fav")
        return

    timestamp = str(int(time.time() * 1000))
    topic = f"tylink/{_device_id}/thing/property/report"
    content = (
        f'{{"msgId":"fav_{timestamp}","time":{timestamp},"data":{{"fav":{{"value":{fav},"time":{timestamp}}}}}}}'
    )
    _mqtt_client.publish(topic, content, qos=1)
    print("上报fav到云端:", content)
