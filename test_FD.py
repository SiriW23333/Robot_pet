from hobot_dnn import pyeasy_dnn as dnn
import cv2
import numpy as np

#create model object
models = dnn.load('FD.bin')


def image_to_nchw():
    # 1. 读取图像（默认为 BGR 格式）
    bgr_image = cv2.imread('./000_0.bmp')
    if bgr_image is None:
        raise FileNotFoundError(f"无法读取图像文件: {src_path}")

    # 2. 转换为 RGB 格式
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 3. 调整维度顺序为 CHW (C, H, W)
    chw_image = np.transpose(rgb_image, (2, 0, 1))  # HWC -> CHW

    # 4. 添加批次维度 (N, C, H, W)
    nchw_image = np.expand_dims(chw_image, axis=0)  # CHW -> NCHW

    # 5. 确保数据类型为 uint8，数值范围 [0, 255]
    assert nchw_image.dtype == np.uint8, "数据类型应为 np.uint8"
    assert np.all((nchw_image >= 0) & (nchw_image <= 255)), "数值范围应为 [0, 255]"

    return nchw_image


img=image_to_nchw()
outputs = models[0].forward(img)
# 查看 outputs 的属性
for i, output in enumerate(outputs):
    print(f"Output {i}:")
    print(f"Data: {output}")
    print(f"{output.buffer}")
    print(f"{output.name}")
    print(f"{output.properties}")