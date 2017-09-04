from operator import itemgetter


import numpy as np


def iou(gt_box, pred_box):
    """Computes intersection over union
    
    Args:
        gt_box(numpy.ndarray): ground truth bounding box in format
            (x_min, y_min, x_max, y_max)
        pred_box(numpy.ndarray): predicted bounding box in format
            (x_min, y_min, x_max, y_max)
    """
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box

    # Compute limits of intersection box
    inter_xmax = min(gt_xmax, pred_xmax)
    inter_xmin = max(gt_xmin, pred_xmin)
    inter_ymax = min(gt_ymax, pred_ymax)
    inter_ymin = max(gt_ymin, pred_ymin)

    # Computer area of intersection
    x_inter_length = max(inter_xmax - inter_xmin, 0)
    y_inter_length = max(inter_ymax - inter_ymin, 0)
    inter_area = x_inter_length * y_inter_length

    # Compute area of union
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    union_area = pred_area + gt_area - inter_area
    return inter_area / union_area


def class_AP(dataset, all_boxes, class_name, iou_thres=0.5):
    class_idx = dataset.classes.index(class_name)
    class_box_infos = all_boxes[class_idx]

    # Extract confidence scores and bbox predictions
    class_confidences = [class_box_info[:, 4]
                         for class_box_info in class_box_infos]
    class_bboxes = [class_box_info[:, 0:4]
                    for class_box_info in class_box_infos]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Go through all images
    for image_idx in range(len(class_bboxes)):
        confidences = class_confidences[image_idx]
        bboxes = class_bboxes[image_idx]
        sorted_idxs = np.argsort(-confidences)

        # Sort confidences and corresponding bbox in detections
        confidences = confidences[sorted_idxs]
        bboxes = bboxes[sorted_idxs]

        # go down detections and mark TPs and FPs
        detected_ids = []  # keeps track of already detected gt bboxes

        # Ground truth information for images
        gt_detections = dataset.annotations[image_idx]['boxes']
        gt_classes = dataset.annotations[image_idx]['gt_classes']
        gt_detections = gt_detections[np.where(gt_classes == class_idx)]
        for conf, bbox in zip(confidences, bboxes):
            detections = []
            # Find best detection in ground truth
            for gt_idx, gt_bbox in enumerate(gt_detections):
                if gt_idx not in detected_ids:
                    bbox_iou = iou(gt_bbox, bbox)
                    if bbox_iou > iou_thres:
                        detections.append((gt_idx, bbox_iou))
            if len(detections) > 0:
                true_positives += 1
                gt_best_detect_id = max(detections, key=itemgetter(1))[0]
                detected_ids.append(gt_best_detect_id)
            else:
                false_positives += 1
        false_negatives += len(gt_detections) - len(detected_ids)
    return true_positives, false_positives, false_negatives
