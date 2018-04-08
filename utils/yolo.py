import cv2
import numpy as np
from .im_transform import imcv2_affine_trans, imcv2_recolor
# from box import BoundBox, box_iou, prob_compare
from utils.nms_wrapper import nms
from utils.cython_yolo import yolo_to_bbox


# This prevents deadlocks in the data loader, caused by
# some incompatibility between pytorch and cv2 multiprocessing.
# See https://github.com/pytorch/pytorch/issues/1355.
cv2.setNumThreads(0)


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def nms_detections(pred_boxes, scores, nms_thresh):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    return keep


def _offset_boxes(boxes, im_shape, scale, offs, flip):
    if len(boxes) == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float)
    boxes *= scale
    boxes[:, 0::2] -= offs[0]
    boxes[:, 1::2] -= offs[1]
    boxes = clip_boxes(boxes, im_shape)

    if flip:
        boxes_x = np.copy(boxes[:, 0])
        boxes[:, 0] = im_shape[1] - boxes[:, 2]
        boxes[:, 2] = im_shape[1] - boxes_x

    return boxes


def preprocess_train(data, size_index):
    im_path, blob, inp_size = data

    boxes, gt_classes = blob['boxes'], blob['gt_classes']

    im = cv2.imread(im_path)
    ori_im = np.copy(im)

    im, trans_param = imcv2_affine_trans(im)
    scale, offs, flip = trans_param
    boxes = _offset_boxes(boxes, im.shape, scale, offs, flip)

    if inp_size is not None and size_index is not None:
        inp_size = inp_size[size_index]
        w, h = inp_size
        boxes[:, 0::2] *= float(w) / im.shape[1]
        boxes[:, 1::2] *= float(h) / im.shape[0]
        im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = imcv2_recolor(im)
    # im /= 255.

    # im = imcv2_recolor(im)
    # h, w = inp_size
    # im = cv2.resize(im, (w, h))
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # im /= 255
    boxes = np.asarray(boxes, dtype=np.int)
    return im, boxes, gt_classes, [], ori_im


def preprocess_test(data, size_index):

    im, _, inp_size = data

    if isinstance(im, str):
        im = cv2.imread(im)
    ori_im = np.copy(im)

    if inp_size is not None and size_index is not None:
        inp_size = inp_size[size_index]
        w, h = inp_size
        im = cv2.resize(im, (w, h))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im / 255.

    return im, [], [], [], ori_im


def postprocess(bbox_pred, iou_pred, prob_pred, im_shape, cfg, thresh=0.05,
                size_index=0):
    """
    bbox_pred: (bsize, HxW, num_anchors, 4)
               ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
    iou_pred: (bsize, HxW, num_anchors, 1)
    prob_pred: (bsize, HxW, num_anchors, num_classes)
    """

    # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
    num_classes = cfg.num_classes
    anchors = cfg.anchors
    W, H = cfg.multi_scale_out_size[size_index]
    assert bbox_pred.shape[0] == 1, 'postprocess only support one image per batch'  # noqa

    bbox_pred = yolo_to_bbox(
        np.ascontiguousarray(bbox_pred, dtype=np.float),
        np.ascontiguousarray(anchors, dtype=np.float),
        H, W)
    bbox_pred = np.reshape(bbox_pred, [-1, 4])
    bbox_pred[:, 0::2] *= float(im_shape[1])
    bbox_pred[:, 1::2] *= float(im_shape[0])
    bbox_pred = bbox_pred.astype(np.int)

    iou_pred = np.reshape(iou_pred, [-1])
    prob_pred = np.reshape(prob_pred, [-1, num_classes])

    cls_inds = np.argmax(prob_pred, axis=1)
    prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
    scores = iou_pred * prob_pred
    # scores = iou_pred
    assert len(scores) == len(bbox_pred), '{}, {}'.format(scores.shape, bbox_pred.shape)
    # threshold
    keep = np.where(scores >= thresh)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # NMS
    keep = np.zeros(len(bbox_pred), dtype=np.int)
    for i in range(num_classes):
        inds = np.where(cls_inds == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bbox_pred[inds]
        c_scores = scores[inds]
        c_keep = nms_detections(c_bboxes, c_scores, 0.3)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    # keep = nms_detections(bbox_pred, scores, 0.3)
    bbox_pred = bbox_pred[keep]
    scores = scores[keep]
    cls_inds = cls_inds[keep]

    # clip
    bbox_pred = clip_boxes(bbox_pred, im_shape)

    return bbox_pred, scores, cls_inds


def _bbox_targets_perimage(im_shape, gt_boxes, cls_inds, dontcare_areas, cfg):
    # num_classes, num_anchors = cfg.num_classes, cfg.num_anchors
    # anchors = cfg.anchors
    H, W = cfg.out_size
    gt_boxes = np.asarray(gt_boxes, dtype=np.float)
    # TODO: dontcare areas
    dontcare_areas = np.asarray(dontcare_areas, dtype=np.float)

    # locate the cell of each gt_boxe
    cell_w = float(im_shape[1]) / W
    cell_h = float(im_shape[0]) / H
    cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5 / cell_w
    cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5 / cell_h
    cell_inds = np.floor(cy) * W + np.floor(cx)
    cell_inds = cell_inds.astype(np.int)

    # [x1, y1, x2, y2],  [class]
    # gt_boxes[:, 0::2] /= im_shape[1]
    # gt_boxes[:, 1::2] /= im_shape[0]
    # gt_boxes[:, 0] = cx - np.floor(cx)
    # gt_boxes[:, 1] = cy - np.floor(cy)
    # gt_boxes[:, 2] = (gt_boxes[:, 2] - gt_boxes[:, 0]) / im_shape[1]
    # gt_boxes[:, 3] = (gt_boxes[:, 3] - gt_boxes[:, 1]) / im_shape[0]

    bbox_target = [[] for _ in range(H*W)]
    cls_target = [[] for _ in range(H*W)]
    for i, ind in enumerate(cell_inds):
        bbox_target[ind].append(gt_boxes[i])
        cls_target[ind].append(cls_inds[i])
    return bbox_target, cls_target


def get_bbox_targets(images, gt_boxes, cls_inds, dontcares, cfg):
    bbox_targets = []
    cls_targets = []
    for i, im in enumerate(images):
        bbox_target, cls_target = _bbox_targets_perimage(im.shape,
                                                         gt_boxes[i],
                                                         cls_inds[i],
                                                         dontcares[i],
                                                         cfg)
        bbox_targets.append(bbox_target)
        cls_targets.append(cls_target)
    return bbox_targets, cls_targets


def draw_detection(im, bboxes, scores, cls_inds, cfg, thr=0.3):
    # draw image
    colors = cfg.colors
    labels = cfg.label_names

    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                    0, 1e-3 * h, colors[cls_indx], thick // 3)

    return imgcv
