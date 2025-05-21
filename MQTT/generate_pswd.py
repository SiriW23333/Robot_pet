import hmac
import hashlib
import time

# 示例参数，请替换为你自己的
def sign_in():
    device_id = "26785d15b5c0e333abuihd "
    device_secret = "1novSR8RExbZDWve"  # 在产品详情页获取
    timestamp = str(int(time.time()))  # 当前时间戳（10位）

    content = f"deviceId={device_id},timestamp={timestamp},secureMode=1,accessType=1"
    password = hmac.new(device_secret.encode(), content.encode(), hashlib.sha256).hexdigest()

    return content,password