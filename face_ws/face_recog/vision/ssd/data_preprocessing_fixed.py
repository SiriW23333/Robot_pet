import cv2
import numpy as np
from typing import List, Tuple, Optional

class Compose:
    """替代PyTorch的Compose"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

# 基础变换类（保持与原始代码相同的接口）
class ConvertFromInts:
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        h, w = img.shape[:2]
        img = cv2.resize(img, (self.size, self.size))
        
        if boxes is not None:
            boxes[:, [0, 2]] *= (self.size / w)
            boxes[:, [1, 3]] *= (self.size / h)
        return img, boxes, labels

class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32) - self.mean
        return img, boxes, labels

class ToPercentCoords:
    def __call__(self, img, boxes=None, labels=None):
        if boxes is not None:
            h, w = img.shape[:2]
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
        return img, boxes, labels

# 地平线专用图像处理（保持原有类定义）
class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # PhotometricDistort(),  # 需要实现
            # RandomSampleCrop_v2(), # 需要实现
            # RandomMirror(),        # 需要实现
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor(),  # 移除PyTorch依赖
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor(),  # 移除PyTorch依赖
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            # ToTensor()  # 移除PyTorch依赖
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image

# 地平线专用扩展（新增）
class HorizonPreprocess:
    """将预处理结果转为地平线BPU需要的NV12格式"""
    @staticmethod
    def to_nv12(img: np.ndarray) -> np.ndarray:
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        return np.frombuffer(yuv, dtype=np.uint8)