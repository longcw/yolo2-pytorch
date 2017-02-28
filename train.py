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
data_loader = VOCDataset(cfg.imdb_name, cfg.year, cfg.DATA_DIR, cfg.batch_size, processes=10, shuffle=True)
print 'start'
for step in range(cfg.max_step):
    batch = data_loader.next_batch()
    print batch['images'][0].shape
    # print step
    # cv2.imshow('test', batch['images'][0])
    #
    # cv2.waitKey(0)

data_loader.close()
