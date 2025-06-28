# -*- coding: utf-8 -*- 
from horizon_tc_ui import HB_ONNXRuntime
import numpy as np
import skimage
from skimage import io
from skimage.transform import resize

# �Զ�������׼������
def your_custom_data_prepare(image_file):
    # skimage��ȡͼƬ���Ѿ���NHWC�Ų�
    image = skimage.img_as_float(io.imread(image_file))
    # ����ȱ����ţ��̱�������256
    image = resize(image, (256, 256), mode='reflect')
    # CenterCrop��ȡ224x224ͼ��
    image = image[16:240, 16:240]  # ������CenterCrop���ֶ�ʵ��
    # ת��ΪBGR˳�򣨼���ԭͼ��RGB��
    image = image[..., ::-1]  # BGR˳��
    # ���ԭģ���� NCHW ���룬����ΪNHWC��ʽ
    image = np.transpose(image, (1, 2, 0))  # HWC -> NHWC
    # ��ͼ�����ֵ��Χ��[0, 1]ת��[0, 255]
    image = image * 255
    # ִ�м�128������������ͨ��ʹ��bgr_128��
    image = image - 128
    # ǿ��ת��Ϊint8����
    image = image.astype(np.int8)
    return image

# ��������Session
def test_inference(model_path, image_path):
    sess = HB_ONNXRuntime(model_file=model_path)

    # ��ȡ����ڵ�����
    input_names = sess.input_names

    # ��ȡ����ڵ�����
    output_names = sess.output_names

    # ׼����������
    feed_dict = {}
    for input_name in input_names:
        feed_dict[input_name] = your_custom_data_prepare(image_path)

    # ����������ȡ���
    outputs = sess.run(output_names, feed_dict, input_offset=128)

    # ��������������Ҫ��������
    print("Model inference results:", outputs)



if __name__ == "__main__":
    model_path = "/root/Robot_pet/face_ws/face_recog/model_output_dir/RF_quantized_model.onnx"  # Replace with your ONNX model path
    image_path = "/root/Robot_pet/face_ws/face_recog/imgs/5.jpg"  # Replace with your input image path

    test_inference(model_path, image_path)
