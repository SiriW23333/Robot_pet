import numpy as np
from collections import namedtuple
from typing import List, Tuple

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])

class SSD:
    def __init__(self, num_classes: int, base_net: List, source_layer_indexes: List[int],
                 extras: List, classification_headers: List, regression_headers: List,
                 is_test=False, config=None, device=None):
        """ÊÊÅäµØÆ½ÏßRDKµÄSSDÊµÏÖ"""
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        
        # µØÆ½ÏßRDK×¨ÓÃ³õÊ¼»¯
        self.model_handle = None  # ÓÃÓÚ´æ´¢µØÆ½ÏßÄ£ÐÍ¾ä±ú
        if is_test:
            self.priors = np.array(config.priors, dtype=np.float32)
            
    def load(self, model_path: str):
        """µØÆ½ÏßRDKÄ£ÐÍ¼ÓÔØ·½·¨"""
        from hobot_dnn import pyeasy_dnn as dnn
        self.model_handle = dnn.load(model_path)
        return self
        
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ç°ÏòÍÆÀí£¨ÊÊÅäµØÆ½ÏßRDK£©"""
        if self.model_handle is None:
            raise RuntimeError("±ØÐëÏÈµ÷ÓÃload()·½·¨¼ÓÔØÄ£ÐÍ")
            
        # È·±£ÊäÈëÊý¾Ý¸ñÊ½ÕýÈ· (NHWC -> NCHW)
        if x.shape[1] == 3:  # Èç¹ûÊäÈëÊÇNCHW¸ñÊ½
            x = np.transpose(x, (0, 2, 3, 1))  # ×ªÎªNHWC
            
        # Ö´ÐÐÍÆÀí
        outputs = self.model_handle[0].forward(x)
        
        # ½âÎöÊä³ö (¸ù¾ÝµØÆ½ÏßÄ£ÐÍÊµ¼ÊÊä³ö½á¹¹µ÷Õû)
        confidences = []
        locations = []
        for output in outputs:
            if output.properties.name.startswith('cls'):
                conf = output.buffer.transpose(0, 2, 3, 1)  # NCHW -> NHWC
                confidences.append(conf.reshape(conf.shape[0], -1, self.num_classes))
            elif output.properties.name.startswith('reg'):
                loc = output.buffer.transpose(0, 2, 3, 1)  # NCHW -> NHWC
                locations.append(loc.reshape(loc.shape[0], -1, 4))
        
        confidences = np.concatenate(confidences, axis=1)
        locations = np.concatenate(locations, axis=1)

        if self.is_test:
            confidences = self.softmax(confidences, axis=2)
            boxes = self.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = self.center_form_to_corner_form(boxes)
            return confidences, boxes
        return confidences, locations

    # ±£³ÖÔ­ÓÐ¹¤¾ß·½·¨²»±ä
    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def convert_locations_to_boxes(self, locations, priors, center_variance, size_variance):
        boxes = np.zeros_like(locations)
        boxes[..., :2] = locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2]
        boxes[..., 2:] = np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        return boxes

    def center_form_to_corner_form(self, boxes):
        return np.concatenate([
            boxes[..., :2] - boxes[..., 2:] / 2,
            boxes[..., :2] + boxes[..., 2:] / 2
        ], axis=-1)

class MatchPrior:
    """±£³ÖÔ­ÓÐÊµÏÖ²»±ä"""
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = self.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        boxes, labels = self.assign_priors(gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold)
        boxes = self.corner_form_to_center_form(boxes)
        locations = self.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels

    def center_form_to_corner_form(self, boxes):
        return np.concatenate([boxes[..., :2] - boxes[..., 2:] / 2,
                               boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    def corner_form_to_center_form(self, boxes):
        return np.concatenate([(boxes[..., :2] + boxes[..., 2:]) / 2,
                               boxes[..., 2:] - boxes[..., :2]], axis=-1)

    def assign_priors(self, gt_boxes, gt_labels, priors, iou_threshold):
        # ¼ò»¯µÄÏÈÑé¿ò·ÖÅäÂß¼­
        ious = self.iou_of(gt_boxes[np.newaxis], priors[:, np.newaxis])
        best_prior_idx = np.argmax(ious, axis=1)
        labels = gt_labels[best_prior_idx]
        labels[np.max(ious, axis=1) < iou_threshold] = 0  # ±³¾°Àà
        return gt_boxes[best_prior_idx], labels

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
        overlap_area = np.prod(np.clip(overlap_right_bottom - overlap_left_top, a_min=0, a_max=None), axis=-1)
        area0 = np.prod(boxes0[..., 2:] - boxes0[..., :2], axis=-1)
        area1 = np.prod(boxes1[..., 2:] - boxes1[..., :2], axis=-1)
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def convert_boxes_to_locations(self, boxes, priors, center_variance, size_variance):
        locations = np.zeros_like(boxes)
        locations[..., :2] = (boxes[..., :2] - priors[..., :2]) / (priors[..., 2:] * center_variance)
        locations[..., 2:] = np.log(boxes[..., 2:] / priors[..., 2:]) / size_variance
        return locations