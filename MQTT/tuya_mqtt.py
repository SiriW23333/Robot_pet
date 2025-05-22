import time
import hmac
import hashlib
import paho.mqtt.client as mqtt

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

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    topic_reply = f"tylink/{userdata['device_id']}/thing/property/set"
    client.subscribe(topic_reply)
    print("Subscribed to:", topic_reply)

def on_message(client, userdata, msg):
    print(f"Topic: {msg.topic}, Payload: {msg.payload.decode()}")

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
    global _mqtt_client, _device_id
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
