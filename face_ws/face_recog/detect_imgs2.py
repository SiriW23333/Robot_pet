"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import sys
import numpy as np
import cv2
from vision.ssd.predictor import Predictor
import hobot_dnn.pyeasy_dnn as dnn
from vision.ssd.ssd import SSD
from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_imgs')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=640, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.6, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="imgs", type=str,
                    help='imgs dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
args = parser.parse_args()
define_img_size(args.input_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

result_path = "./detect_imgs_results"
label_path = "./models/voc-model-labels.txt"
test_device = args.test_device

class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=test_device)
    net.load(model_path)
elif args.net_type == 'RFB':
    model_path = "/root/Robot_pet/face_ws/face_recog/FD.bin"
    if not os.path.exists(model_path):
      print(f"Error: Model path {model_path} does not exist!")
      sys.exit(1)

    model = dnn.Model(model_path)
    if model is None:
        print("Error: Failed to load the model!")
        sys.exit(1)
    # model_path = "models/pretrained/version-RFB-640.pth"
    #net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    model=dnn.Model(model_path)
    predictor = create_Mb_Tiny_RFB_fd_predictor(model, candidate_size=args.candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
#net.load(model_path)

listdir = os.listdir(args.path)
listdir.sort()  # 排序，保证第一张图顺序稳定

if len(listdir) == 0:
    print("No images found in the directory.")
    sys.exit(1)

file_path = listdir[0]  # 只取第一张图片
img_path = os.path.join(args.path, file_path)
orig_image = cv2.imread(img_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}, dtype: {image.dtype}")
# resize到模型输入大小 (这里假设640x640,需根据模型调整)
image_resized = cv2.resize(image, (640, 640))
# 颜色通道 (根据模型调整是BGR还是RGB)
image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
# 转为float32
image_resized = image_resized.astype('float32')
# 若模型要求归一化:
image_resized = (image_resized - 127.5) / 128.0  # 示例归一化,需根据模型调整
# 调整为CHW格式
image_resized = np.transpose(image_resized, (2, 0, 1))
tensor_input = dnn.create_tensor(image)
boxes, labels, probs = predictor.predict(tensor_input)

print(f"Processing image: {file_path}")
print(f"Found {boxes.size(0)} faces")

    # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
for i in range(boxes.size(0)):
    box = boxes[i, :]
    pt1 = (int(box[0].item()), int(box[1].item()))
    pt2 = (int(box[2].item()), int(box[3].item()))
    cv2.rectangle(orig_image, pt1, pt2, (0, 0, 255), 2)
    label = f"{probs[i]:.2f}"
cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

if not os.path.exists(result_path):
    os.makedirs(result_path)
cv2.imwrite(os.path.join(result_path, file_path), orig_image)
print(f"Output saved to {os.path.join(result_path, file_path)}")

# if not os.path.exists(result_path):
#     os.makedirs(result_path)
# listdir = os.listdir(args.path)
# sum = 0
# for file_path in listdir:
#     img_path = os.path.join(args.path, file_path)
#     orig_image = cv2.imread(img_path)
#     image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#     boxes, labels, probs = predictor.predict(image, args.candidate_size / 2, args.threshold)
#     sum += boxes.size(0)
#     for i in range(boxes.size(0)):
#         box = boxes[i, :]
#         cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
#         # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
#         label = f"{probs[i]:.2f}"
#         # cv2.putText(orig_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.imwrite(os.path.join(result_path, file_path), orig_image)
#     print(f"Found {len(probs)} faces. The output image is {result_path}")
# print(sum)
