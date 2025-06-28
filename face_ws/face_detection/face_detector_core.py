import cv2
import numpy as np
import onnxruntime
from utils.utils import letterbox_image, preprocess_input
from utils.anchors import Anchors
from utils.config import cfg_mnet
import torch
import os

class FaceDetector:
    def __init__(self, onnx_path, confidence=0.6):
      
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (640, 640)
        self.confidence = confidence
        self.nms_iou = 0.45
        self.anchors = Anchors(cfg_mnet, image_size=self.input_shape).get_anchors()

    def detect_faces(self, image):
        """
        检测人脸
        
        Args:
            image: 输入图像 (BGR格式的numpy数组)
            
        Returns:
            list: 人脸检测结果列表，每个元素为 (x1, y1, x2, y2, confidence)
                 x1, y1: 左上角坐标
                 x2, y2: 右下角坐标  
                 confidence: 置信度 (0-1之间的浮点数)
        """
        # 图像预处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_height, im_width = image.shape[:2]
        
        # resize和归一化
        processed_img = letterbox_image(image_rgb, [self.input_shape[1], self.input_shape[0]])
        processed_img = preprocess_input(processed_img)
        processed_img = np.transpose(processed_img, (2, 0, 1))
        processed_img = np.expand_dims(processed_img, axis=0)
        
        # 模型推理
        outputs = self.session.run(None, {self.input_name: processed_img.astype(np.float32)})
        
        # 解析输出
        loc, conf, _ = outputs
        loc = loc[0]
        conf = conf[0]
        
        # 过滤低置信度检测
        scores = conf[:, 1]  # 人脸类别的置信度
        
        # 应用sigmoid激活函数将logits转换为概率
        scores = 1.0 / (1.0 + np.exp(-scores))
        
        mask = scores > self.confidence
        
        if not np.any(mask):
            return []
        
        # anchor解码
        from utils.utils_bbox import decode
        variance = [0.1, 0.2]
        
        loc_filtered = loc[mask]
        scores_filtered = scores[mask]
        anchors_filtered = self.anchors[mask]
        
        # 解码边界框
        boxes = decode(torch.from_numpy(loc_filtered), anchors_filtered, variance)
        boxes = boxes.cpu().numpy()
        
        # 组合结果
        result = np.concatenate([boxes, scores_filtered.reshape(-1, 1)], axis=1)
        
        # NMS去重
        from torchvision.ops import nms
        keep = nms(torch.from_numpy(result[:, :4]), 
                  torch.from_numpy(result[:, 4]), 
                  self.nms_iou)
        
        if len(keep) == 0:
            return []
        
        result = result[keep.cpu().numpy()]
        
        # 手动进行坐标还原（letterbox逆变换）
        input_shape = np.array([self.input_shape[0], self.input_shape[1]])
        image_shape = np.array([im_height, im_width])
        
        # 计算letterbox的缩放和偏移
        new_shape = image_shape * np.min(input_shape / image_shape)
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        
        scale_for_boxes = [scale[1], scale[0], scale[1], scale[0]]
        offset_for_boxes = [offset[1], offset[0], offset[1], offset[0]]
        
        # 应用逆变换：先减去offset，再乘以scale
        result[:, :4] = (result[:, :4] - np.array(offset_for_boxes)) * np.array(scale_for_boxes)
        
        # 缩放到原图尺寸
        scale_to_image = [im_width, im_height, im_width, im_height]
        result[:, :4] = result[:, :4] * scale_to_image
        
        # 格式化输出
        faces = []
        for face in result:
            x1, y1, x2, y2, confidence = face
            faces.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
        
        return faces
