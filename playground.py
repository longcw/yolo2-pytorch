from __future__ import print_function, division
import os
import csv
from skimage import io, transform
import pandas as pd
import torch
import numpy as np
import cv2
from utils import im_transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from lisa_dataset import LISADataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from darknet import Darknet19

images_file = 'datasets/LISA/allAnnotations.csv'
tags_file = 'datasets/LISA/tags.csv'
root_dir = 'datasets/LISA'
ds = LISADataset(images_file, tags_file, root_dir)

net = Darknet19()
net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
#net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
num_epochs = 25

lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
index = 0
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
batch_size = 16
for i in range(0, num_epochs):
    ds.reset()
    cnt = 0
    print('Epoch #: ', i)
    while(ds.hasMoreImages()):
    # batch
    
        images, labels, classes = ds.getBatch(batch_size = batch_size)
        im_data = torch.autograd.Variable(images).cuda()
        dont_care=np.array([[], [], [], [], [], [], [], [],
                           [], [], [], [], [], [], [], []])
        # forward

        try:
            net(im_data, labels, classes, dont_care)
        except IndexError:
            continue
            
            
        cnt+=batch_size
        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
        iou_loss += net.iou_loss.data.cpu().numpy()[0]
        cls_loss += net.cls_loss.data.cpu().numpy()[0]
        train_loss += loss.data.cpu().numpy()[0]
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print('average loss: ', train_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
save_name = os.path.join('models', '{}_{}.h5'.format('retrained_yolo', num_epochs))
net_utils.save_net(save_name, net)
print('save model: {}'.format(save_name))

