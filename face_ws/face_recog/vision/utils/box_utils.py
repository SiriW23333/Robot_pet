import math
import numpy as np

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    
    print("priors nums:{}".format(len(priors)))
    priors = np.array(priors, dtype=np.float32)
    if clamp:
        priors = np.clip(priors, 0.0, 1.0)
    return priors

def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    if priors.ndim + 1 == locations.ndim:
        priors = np.expand_dims(priors, axis=0)
    
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=locations.ndim - 1)

def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    if center_form_priors.ndim + 1 == center_form_boxes.ndim:
        center_form_priors = np.expand_dims(center_form_priors, axis=0)
    
    return np.concatenate([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        np.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], axis=center_form_boxes.ndim - 1)

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def assign_priors(gt_boxes, gt_labels, corner_form_priors, iou_threshold):
    ious = iou_of(np.expand_dims(gt_boxes, axis=0), 
                 np.expand_dims(corner_form_priors, axis=1))
    
    best_target_per_prior = np.max(ious, axis=1)
    best_target_per_prior_index = np.argmax(ious, axis=1)
    
    best_prior_per_target = np.max(ious, axis=0)
    best_prior_per_target_index = np.argmax(ious, axis=0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels

def hard_negative_mining(loss, labels, neg_pos_ratio):
    pos_mask = labels > 0
    num_pos = np.sum(pos_mask.astype(np.int64), axis=1, keepdims=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -np.inf
    indexes = np.argsort(-loss, axis=1)  # Ωµ–Ú≈≈–Ú
    orders = np.argsort(indexes, axis=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

def center_form_to_corner_form(locations):
    return np.concatenate([
        locations[..., :2] - locations[..., 2:] / 2,
        locations[..., :2] + locations[..., 2:] / 2
    ], axis=locations.ndim - 1)

def corner_form_to_center_form(boxes):
    return np.concatenate([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], axis=boxes.ndim - 1)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(-scores)[:candidate_size]
    
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
            
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0)
            )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size)

def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    picked_box_scores = []
    box_scores = box_scores.copy()
    
    while box_scores.shape[0] > 0:
        max_score_index = np.argmax(box_scores[:, 4])
        cur_box_prob = box_scores[max_score_index, :].copy()
        picked_box_scores.append(cur_box_prob)
        
        if (top_k > 0 and len(picked_box_scores) == top_k) or box_scores.shape[0] == 1:
            break
            
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        
        ious = iou_of(np.expand_dims(cur_box, axis=0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * np.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    
    if len(picked_box_scores) > 0:
        return np.stack(picked_box_scores)
    else:
        return np.array([])