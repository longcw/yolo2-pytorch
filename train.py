import os
import cv2
import torch
import numpy as np
from torch.multiprocessing import Pool

from darknet import Darknet19

from datasets.pascal_voc import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


# data loader
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True, dst_size=cfg.inp_size)
print 'start'
for step in range(cfg.max_step):
    batch = imdb.next_batch()

    im = batch['images'][0]
    gt_boxes = batch['gt_boxes'][0]
    cls_inds = batch['gt_classes'][0]
    orgin_im = batch['origin_im'][0]

    yolo_utils.anchor_target_one_image(im.shape, gt_boxes, batch['dontcare'], cfg)

    print gt_boxes
    im2show = yolo_utils.draw_detection(im, gt_boxes, np.ones(len(cls_inds)), cls_inds, cfg)

    cv2.imshow('train', im2show)
    cv2.waitKey(1)

    # print batch['gt_boxes']
    # print batch['gt_classes']
    # print batch['images'][0].shape
    # print step
    # cv2.imshow('test', batch['images'][0])
    #
    # cv2.waitKey(20)

imdb.close()
