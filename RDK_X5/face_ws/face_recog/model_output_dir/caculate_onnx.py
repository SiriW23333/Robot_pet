# -*- coding: utf-8 -*- 
from horizon_tc_ui import HB_ONNXRuntime
import numpy as np
import skimage
from skimage import io
from skimage.transform import resize

# 自定义数据准备函数
def your_custom_data_prepare(image_file):
    # skimage读取图片，已经是NHWC排布
    image = skimage.img_as_float(io.imread(image_file))
    # 长宽等比缩放，短边缩放至256
    image = resize(image, (256, 256), mode='reflect')
    # CenterCrop获取224x224图像
    image = image[16:240, 16:240]  # 这里是CenterCrop的手动实现
    # 转换为BGR顺序（假设原图是RGB）
    image = image[..., ::-1]  # BGR顺序
    # 如果原模型是 NCHW 输入，调整为NHWC格式
    image = np.transpose(image, (1, 2, 0))  # HWC -> NHWC
    # 将图像的数值范围从[0, 1]转到[0, 255]
    image = image * 255
    # 执行减128操作（量化后通常使用bgr_128）
    image = image - 128
    # 强制转换为int8类型
    image = image.astype(np.int8)
    return image

# 创建推理Session
def test_inference(model_path, image_path):
    sess = HB_ONNXRuntime(model_file=model_path)

    # 获取输入节点名称
    input_names = sess.input_names

    # 获取输出节点名称
    output_names = sess.output_names

    # 准备输入数据
    feed_dict = {}
    for input_name in input_names:
        feed_dict[input_name] = your_custom_data_prepare(image_path)

    # 进行推理，获取输出
    outputs = sess.run(output_names, feed_dict, input_offset=128)

    # 输出结果（根据需要处理结果）
    print("Model inference results:", outputs)



if __name__ == "__main__":
    model_path = "/root/Robot_pet/face_ws/face_recog/model_output_dir/RF_quantized_model.onnx"  # Replace with your ONNX model path
    image_path = "/root/Robot_pet/face_ws/face_recog/imgs/5.jpg"  # Replace with your input image path

    test_inference(model_path, image_path)
