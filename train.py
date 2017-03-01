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
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True, dst_size=cfg.inp_size)
print('load data succ...')

net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    # if remove_all_log:
    #     cc.remove_all_experiments()
    if start_epoch == 0:
        exp = cc.create_experiment(cfg.exp_name)
    else:
        exp = cc.open_experiment(cfg.exp_name)

train_loss = 0
t = Timer()
for step in range(start_epoch * imdb.batch_per_epoch, cfg.max_epoch * imdb.batch_per_epoch):
    t.tic()
    # batch
    batch = imdb.next_batch()
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']

    # forward
    im_data = net_utils.np_to_variable(im, is_cuda=True, volatile=False).permute(0, 3, 1, 2)
    net(im_data, gt_boxes, gt_classes, dontcare)

    # backward
    loss = net.loss
    train_loss += loss.data.cpu().numpy()[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    duration = t.toc()
    if step % 10 == 0:
        print('step: %d, loss: %.3f, %.2f s/batch' % (step, train_loss/10, duration))
        train_loss = 0
        t.clear()

    if step > 0 and (step % imdb.batch_per_epoch == 0):
        if imdb.epoch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

        save_name = os.path.join(cfg.train_output_dir, '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
        net_utils.save_net(save_name, net)
        print('save model: {}'.format(save_name))


imdb.close()
