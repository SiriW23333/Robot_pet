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
        # MQTT username
        self.username = f"{device_id}|signMethod=hmacSha256,timestamp={timestamp},secureMode=1,accessType=1"
        # MQTT clientId
        self.client_id = f"tuyalink_{device_id}"
        plain_passwd = f"deviceId={device_id},timestamp={timestamp},secureMode=1,accessType=1"
        print("plainPasswd=", plain_passwd)
        # MQTT password
        self.password = self.hmac_sha256(plain_passwd, device_secret)

    @staticmethod
    def hmac_sha256(plain_text, key):
        # Zero filling less than 64 characters
        digest = hmac.new(key.encode(), plain_text.encode(), hashlib.sha256).hexdigest()
        return digest.zfill(64)

# ------------------- MQTT 连接与发布/订阅示例 -------------------

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    topic_reply = f"tylink/{userdata['device_id']}/thing/property/set"
    client.subscribe(topic_reply)
    print("Subscribed to:", topic_reply)

def on_message(client, userdata, msg):
    print(f"Topic: {msg.topic}, Payload: {msg.payload.decode()}")

if __name__ == "__main__":
    # 使用你提供的设备信息
    product_key = "pgvauagmoqlb5ymz"
    device_id = "26785d15b5c0e333abuihd"
    device_secret = "1novSR8RExbZDWve"
    broker = "m1.tuyacn.com"
    port = 8883

    sign = TuyaMqttSign()
    sign.calculate(product_key, device_id, device_secret)

    client = mqtt.Client(client_id=sign.client_id, userdata={"device_id": device_id})
    client.username_pw_set(sign.username, sign.password)
    # 如有证书，取消注释并指定路径
    # client.tls_set(ca_certs="iot-device.cer")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(broker, port, 60)

    # 发布属性上报
    timestamp = str(int(time.time() * 1000))
    topic = f"tylink/{device_id}/thing/property/report"
    content = f'{{"msgId":"45lkj3551234002","time":{timestamp},"data":{{"switch_led_1":{{"value":true,"time":{timestamp}}}}}}}'
    client.publish(topic, content, qos=1)
    print("Published to:", topic)

    client.loop_forever()