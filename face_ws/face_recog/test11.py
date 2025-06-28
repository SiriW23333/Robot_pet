# -*- coding: utf-8 -*-
import argparse
import os
import sys
import numpy as np
import cv2
import hobot_dnn.pyeasy_dnn as dnn

# NMS 算法，使用与上面的代码一致
def nms(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# 将 center-form 转换为 corner-form
def center_form_to_corner_form(locations):
    """Convert center-form boxes to corner-form boxes."""
    return np.concatenate([locations[:, :2] - locations[:, 2:] / 2,
                           locations[:, :2] + locations[:, 2:] / 2], axis=-1)

# 调整输入图像尺寸并保持原图比例
def letterbox_image(image, target_size):
    src_h, src_w = image.shape[:2]
    target_w, target_h = target_size

    # 计算缩放比例
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建填充背景
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]  # 填充为黑色
    )
    return padded_image, scale, left, top

def main():
    parser = argparse.ArgumentParser(description='Batch detect images using Horizon dnnpy.')
    parser.add_argument('--input_size', default=640, type=int, help='Model input size (width=height)')
    parser.add_argument('--threshold', default=0.6, type=float, help='Detection score threshold')
    parser.add_argument('--path', default="imgs", type=str, help='Input image directory or image file')
    parser.add_argument('--model_path', default="/root/Robot_pet/face_ws/face_recog/FD.bin", type=str, help='Path to .bin model')
    parser.add_argument('--result_path', default="./detect_imgs_results", type=str, help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        sys.exit(1)

    models = dnn.load(args.model_path)
    if not models or len(models) == 0:
        print("Error: Failed to load model or model list is empty.")
        sys.exit(1)
    model = models[0]

    print("Model name:", model.name)
    print("Model inputs info:")
    for i, inp in enumerate(model.inputs):
        prop = inp.properties
        print(f" Input {i}: name={inp.name}, shape={prop.shape}, dtype={prop.dtype}, layout={prop.layout}")

    print("Model outputs info:")
    for i, out in enumerate(model.outputs):
        prop = out.properties
        print(f" Output {i}: name={out.name}, shape={prop.shape}, dtype={prop.dtype}, layout={prop.layout}")

    print("Model loaded successfully.")
    input = models[0].inputs[0]

    # 获取待检测图片列表
    image_files = []
    if os.path.isfile(args.path):
        image_files = [os.path.basename(args.path)]
        args.path = os.path.dirname(args.path)
    elif os.path.isdir(args.path):
        image_files = [f for f in os.listdir(args.path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    else:
        print(f"Error: Path {args.path} is not a valid file or directory.")
        sys.exit(1)

    image_files.sort()
    if len(image_files) == 0:
        print("No images found in the directory.")
        sys.exit(1)

    os.makedirs(args.result_path, exist_ok=True)

    for idx, file_name in enumerate(image_files):
        img_path = os.path.join(args.path, file_name)
        orig_image = cv2.imread(img_path)
        if orig_image is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue

        input_w = 640
        input_h = 480
        image_resized = cv2.resize(orig_image, (input_w, input_h))  # (width, height)

        # 转成uint8，NCHW格式
        image_nchw = np.transpose(image_resized, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(image_nchw, axis=0).astype(np.uint8)  # (1,3,480,640)

        # 推理
        outputs = model.forward(input_data)
        if len(outputs) < 2:
            print(f"Warning: Model output for {file_name} is invalid.")
            continue

        confidences = np.squeeze(outputs[0].buffer, axis=(0,3))  # (17640, 2)
        boxes = np.squeeze(outputs[1].buffer, axis=(0,3))        # (17640, 4)

        detected_boxes = []
        scores = []
        boxes_list = []

        for i in range(confidences.shape[0]):
            score = confidences[i][1]  # 正类概率
            if score >= args.threshold:
                box = boxes[i]
                print(box)
                # 归一化坐标
                x1 = box[0] * input_w
                y1 = box[1] * input_h
                x2 = box[2] * input_w
                y2 = box[3] * input_h

                # 映射回原图
                x1 = int(round(box[0] * orig_image.shape[1]))
                y1 = int(round(box[1] * orig_image.shape[0]))
                x2 = int(round(box[2] * orig_image.shape[1]))
                y2 = int(round(box[3] * orig_image.shape[0]))

                # 确保坐标在范围内
                x1 = max(0, min(x1, orig_image.shape[1] - 1))
                y1 = max(0, min(y1, orig_image.shape[0] - 1))
                x2 = max(0, min(x2, orig_image.shape[1] - 1))
                y2 = max(0, min(y2, orig_image.shape[0] - 1))

                boxes_list.append([x1, y1, x2, y2])
                scores.append(score)

        if len(boxes_list) == 0:
            print(f"No detections found in {file_name}.")
            continue

        boxes_np = np.array(boxes_list)
        scores_np = np.array(scores)

        # NMS 处理
        keep_idx = nms(boxes_np, scores_np, iou_threshold=0.5)

        for idx in keep_idx:
            x1, y1, x2, y2 = boxes_np[idx]
            score = scores_np[idx]
            detected_boxes.append((x1, y1, x2, y2, score))

        # 画框和分数
        for (x1, y1, x2, y2, score) in detected_boxes:
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{score:.2f}"
            cv2.putText(orig_image, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(orig_image, f"{len(detected_boxes)} faces", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        output_path = os.path.join(args.result_path, file_name)
        cv2.imwrite(output_path, orig_image)
        print(f"Processed {file_name}: {len(detected_boxes)} faces detected. Output saved to {output_path}")

if __name__ == '__main__':
    main()
