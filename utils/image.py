import cv2
import os
import numpy as np
from im_transform import imcv2_affine_trans, imcv2_recolor
from box import BoundBox, box_iou, prob_compare


def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def expit(x):
    return 1. / (1. + np.exp(-x))


def preprocess(im, cfg, allobj=None):
    """
    Takes an image, return it as a numpy tensor that is readily
    to be fed into tfnet. If there is an accompanied annotation (allobj),
    meaning this preprocessing is serving the train process, then this
    image will be transformed with random noise to augment training data,
    using scale, translation, flipping and recolor. The accompanied
    parsed annotation (allobj) will also be modified accordingly.
    """
    if isinstance(im, (str, unicode)):
        im = cv2.imread(im)

    if allobj is not None:  # in training mode
        result = imcv2_affine_trans(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip:
                continue
            obj_1_ = obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        im = imcv2_recolor(im)

    h, w = cfg.inp_size
    imsz = cv2.resize(im, (h, w))
    # imsz = imsz[:, :, ::-1]
    imsz = cv2.cvtColor(imsz, cv2.COLOR_BGR2RGB)
    imsz = imsz / 255.

    return imsz


def postprocess(net_out, im, cfg):
    """
    Takes net output, draw net_out, save to disk
    net_out: [c, h, w]
    """
    # meta

    threshold = cfg.thresh
    C, B = cfg.classes, len(cfg.anchors)
    anchors = cfg.anchors
    _, H, W = net_out.shape
    net_out = np.transpose(net_out, [1, 2, 0]).reshape([H, W, B, -1])
    # print net_out
    # print net_out.shape

    boxes = list()
    for row in range(H):
        for col in range(W):
            for b in range(B):
                bx = BoundBox(C)
                bx.x, bx.y, bx.w, bx.h, bx.c = net_out[row, col, b, :5]
                bx.c = expit(bx.c)
                bx.x = (col + expit(bx.x)) / W
                bx.y = (row + expit(bx.y)) / H
                bx.w = np.exp(bx.w) * anchors[b][0] / W
                bx.h = np.exp(bx.h) * anchors[b][1] / H

                classes = net_out[row, col, b, 5:]
                bx.probs = _softmax(classes) * bx.c
                bx.probs *= bx.probs > threshold
                boxes.append(bx)

    # non max suppress boxes
    for c in range(C):
        for i in range(len(boxes)):
            boxes[i].class_num = c
        boxes = sorted(boxes, key=prob_compare)
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.probs[c] == 0: continue
            for j in range(i + 1, len(boxes)):
                boxj = boxes[j]
                if box_iou(boxi, boxj) >= .4:
                    boxes[j].probs[c] = 0.

    colors = cfg.colors
    labels = cfg.label_names
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else:
        imgcv = im
    h, w, _ = imgcv.shape
    for b in boxes:
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = 'object' * int(C < 2)
        label += labels[max_indx] * int(C > 1)
        if max_prob > threshold:
            left = int((b.x - b.w / 2.) * w)
            right = int((b.x + b.w / 2.) * w)
            top = int((b.y - b.h / 2.) * h)
            bot = int((b.y + b.h / 2.) * h)
            if left < 0:  left = 0
            if right > w - 1: right = w - 1
            if top < 0:   top = 0
            if bot > h - 1:   bot = h - 1
            thick = int((h + w) / 300)
            cv2.rectangle(imgcv,
                          (left, top), (right, bot),
                          colors[max_indx], thick)
            mess = '{}'.format(label)
            cv2.putText(imgcv, mess, (left, top - 12),
                        0, 1e-3 * h, colors[max_indx], thick // 3)

    # outfolder = os.path.join('demo', 'out')
    # if not os.path.exists(outfolder):
    #     os.mkdir(outfolder)
    # img_name = os.path.join(outfolder, im.split('/')[-1])
    # cv2.imwrite(img_name, imgcv)
    return imgcv