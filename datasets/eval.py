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
    x_inter_length = max(inter_xmax - inter_xmin + 1, 0)
    y_inter_length = max(inter_ymax - inter_ymin + 1, 0)
    inter_area = x_inter_length * y_inter_length

    # Compute area of union
    gt_area = (gt_xmax - gt_xmin + 1) * (gt_ymax - gt_ymin + 1)
    pred_area = (pred_xmax - pred_xmin + 1) * (pred_ymax - pred_ymin + 1)
    union_area = pred_area + gt_area - inter_area
    return inter_area / union_area


def class_AP(dataset, all_boxes, class_name, iou_thres=0.5):
    """For each image go through detected bounding boxes by decreasing
    order of confidence, for each detected bounding box, it is matches a ground
    truth bounding box for the ground truth bounding box that has maximun iou
    score among the ones that have a iou score abose iou_thres.
    If no ground truth bounding box matches this criterion,
    the detection is a false positive

    Args:
        all_boxes: list of lists of bounding boxes
            (all_boxes[class_idx][image_idx] contains a numpy array of size
            (n,5) where n is the number of bounding boxes and the five values
            are (x_min, y_min, x_max, y_max, confidence)
        class_name(str): name of class for which to compute the average
            precision
        iou_thres(float): minimum iou score for a bounding box to be considered
            as a valid detection

    Returns:
        precision(list): precision (tp/(tp + fp))scores when positive threshold
            is of decreasing confidence
        recall(list): recall (tp/p) scores when positive threshold is of
            decreasing confidence
    """
    class_idx = dataset.classes.index(class_name)
    class_box_infos = all_boxes[class_idx]

    # Extract confidence scores and bbox predictions
    class_confidences = [class_box_info[:, 4]
                         for class_box_info in class_box_infos]
    class_bboxes = [class_box_info[:, 0:4]
                    for class_box_info in class_box_infos]

    true_positive_conf = []
    false_positive_conf = []
    positive_count = 0

    # Go through all images
    for image_idx in range(len(class_bboxes)):
        confidences = class_confidences[image_idx]
        bboxes = class_bboxes[image_idx]
        sorted_idxs = np.argsort(-confidences)

        # Sort confidences and corresponding bbox in detections
        confidences = confidences[sorted_idxs]
        bboxes = bboxes[sorted_idxs]

        detected_ids = []  # keeps track of already detected gt bboxes

        # Ground truth information for image
        gt_detections = dataset.annotations[image_idx]['boxes']

        # Only focus on ground truth samples of the given class
        gt_classes = dataset.annotations[image_idx]['gt_classes']
        gt_detections = gt_detections[np.where(gt_classes == class_idx)]

        # Count total number of positive detections
        positive_count += len(gt_detections)

        # Go down sorted detections and mark TPs and FPs
        for conf, bbox in zip(confidences, bboxes):
            detections = []
            # Find best detection in ground truth
            for gt_idx, gt_bbox in enumerate(gt_detections):
                if gt_idx not in detected_ids:
                    bbox_iou = iou(gt_bbox, bbox)
                    if bbox_iou > iou_thres:
                        # Bounding boxes that match iou criterion
                        detections.append((gt_idx, bbox_iou))
            if len(detections) > 0:
                true_positive_conf.append((1, conf))
                false_positive_conf.append((0, conf))
                gt_best_detect_id = max(detections, key=itemgetter(1))[0]
                detected_ids.append(gt_best_detect_id)
            else:
                false_positive_conf.append((1, conf))
                true_positive_conf.append((0, conf))

    # List true positives and false positives by decreasing confidence
    true_pos_conf_sort = sorted(true_positive_conf, key=itemgetter(1),
                                reverse=True)
    false_pos_conf_sort = sorted(false_positive_conf, key=itemgetter(1),
                                 reverse=True)

    # Extract tp and fp values for decreasing confidence
    false_pos_sort = [flag for flag, conf in false_pos_conf_sort]
    true_pos_sort = [flag for flag, conf in true_pos_conf_sort]

    cum_true_pos = np.cumsum(true_pos_sort)
    cum_false_pos = np.cumsum(false_pos_sort)

    recall = cum_true_pos / positive_count
    precision = cum_true_pos / (cum_true_pos + cum_false_pos)
    return precision, recall
