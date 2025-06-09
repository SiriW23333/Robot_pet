import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer

# Numpy实现的NMS
def numpy_nms(boxes, iou_threshold=0.3, top_k=-1):
    if boxes.shape[0] == 0:
        return boxes
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    indices = scores.argsort()[::-1]
    keep = []
    while indices.size > 0:
        current = indices[0]
        keep.append(current)
        if 0 < top_k == len(keep):
            break
        rest = indices[1:]

        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area1 = (x2[current] - x1[current]) * (y2[current] - y1[current])
        area2 = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
        iou = inter / (area1 + area2 - inter + 1e-6)

        indices = rest[iou <= iou_threshold]
    return boxes[keep]


class Predictor:
    def __init__(self, model, size, mean=0.0, std=1.0, iou_threshold=0.3,
                 filter_threshold=0.01, candidate_size=200, nms_method=None, sigma=0.5):
        self.model = model
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method  # 可忽略，地瓜上只用NMS
        self.sigma = sigma
        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        height, width, _ = image.shape
        image = self.transform(image)  # 返回的是归一化、resize后、float32格式
        image = np.expand_dims(image, axis=0)  # (1, H, W, C) 或 (1, C, H, W)

        # 模型推理
        self.timer.start()
        outputs = self.model.forward(image)
        print("Inference time: ", self.timer.end())

        # 根据输出顺序来确定哪一个是 boxes 哪一个是 scores
        out0 = outputs[0].buffer
        out1 = outputs[1].buffer
        scores = out0 if out0.shape[1] > 4 else out1  # [N, num_classes]
        boxes = out1 if out0.shape[1] > 4 else out0   # [N, 4]

        if prob_threshold is None:
            prob_threshold = self.filter_threshold

        picked_box_probs = []
        picked_labels = []

        num_classes = scores.shape[1]
        for class_index in range(1, num_classes):  # 跳过背景
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            if np.sum(mask) == 0:
                continue
            subset_boxes = boxes[mask]
            subset_scores = probs[mask].reshape(-1, 1)
            box_probs = np.concatenate([subset_boxes, subset_scores], axis=1)

            box_probs = numpy_nms(box_probs, iou_threshold=self.iou_threshold, top_k=top_k)
            if box_probs.shape[0] > 0:
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])

        picked_box_probs = np.concatenate(picked_box_probs, axis=0)
        picked_labels = np.array(picked_labels)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        return picked_box_probs[:, :4], picked_labels, picked_box_probs[:, 4]
