import os
import torch
import datetime
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

from darknet import Darknet19

from cfgs import config as cfg
from datasets.lisa_hd import LISADataset
from datasets.egohands import EgoHandDataset
import utils.network as net_utils
from utils.timer import Timer

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


train_transform = transforms.ToTensor()

transform_params = {
    'shape': (416, 416),
    'jitter': 0.1,
    'hue': 0.1,
    'saturation': 1.5,
    'exposure': 1.5}


def collate_img(data_list):
    imgs = [data_item[0] for data_item in data_list]
    imgs = torch.stack(imgs, 0)
    origin_imgs = [data_item[1] for data_item in data_list]
    targets = {}
    targets['boxes'] = [data_item[2]['boxes'] for data_item in data_list]
    targets['gt_classes'] = [data_item[2]['gt_classes']
                             for data_item in data_list]
    return imgs, origin_imgs, targets


# train_batch_size = cfg.train_batch_size
train_batch_size = 16

# Initialize dataset
if cfg.dataset_name == 'lisa':
    imdb = LISADataset('train', cfg.DATA_DIR, transform=train_transform,
                       transform_params=transform_params, use_cache=False)
elif cfg.dataset_name == 'egohands':
    imdb = EgoHandDataset('train', cfg.DATA_DIR, transform=train_transform,
                          transform_params=transform_params, use_cache=False)
else:
    raise ValueError('dataset name {} not recognized'.format(cfg.dataset_name))
print('load data succ...')

batch_per_epoch = int(len(imdb) / train_batch_size)

# Initialize dataloader
dataloader = DataLoader(imdb, shuffle=True, batch_size=train_batch_size,
                        num_workers=0, collate_fn=collate_img)

# Initialize network
net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                            momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
count = 0
t = Timer()

for epoch in range(cfg.max_epoch):
    for step, (img, original_img, targets) in enumerate(dataloader):
        t.tic()
        gt_boxes = targets['boxes']
        gt_classes = targets['gt_classes']

        # Prepare torch variable
        img_var = Variable(img.type(torch.FloatTensor))
        img_var = img_var.cuda()
        net(img_var, gt_boxes, gt_classes)

        # backward
        loss = net.loss
        bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
        iou_loss += net.iou_loss.data.cpu().numpy()[0]
        cls_loss += net.cls_loss.data.cpu().numpy()[0]
        train_loss += loss.data.cpu().numpy()[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        duration = t.toc()
        if step % cfg.disp_interval == 0:
            train_loss /= count
            bbox_loss /= count
            iou_loss /= count
            cls_loss /= count

            elapsed_time = str(datetime.timedelta(seconds=int((batch_per_epoch
                                                               - step) * duration)))
            print('epoch {}[{}/{}], loss: {:.3f}, bbox_loss: {:.3f}, iou_loss: {:.3f}, cls_loss: {:.3f} ({:.3f} s/batch, rest:{})'.format(epoch, step, batch_per_epoch,
                                                                                                                                          train_loss, bbox_loss, iou_loss,
                                                                                                                                          cls_loss, duration, elapsed_time))

            train_loss = 0
            bbox_loss, iou_loss, cls_loss = 0., 0., 0.
            count = 0
            t.clear()

    if epoch in cfg.lr_decay_epochs:
        lr *= cfg.lr_decay
        print('Using new learning rate {}'.format(lr))
        optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay)

    save_name = os.path.join(cfg.train_output_dir,
                             '{}_{}.h5'.format(cfg.exp_name,
                                               epoch))
    net_utils.save_net(save_name, net)
    print('save model: {}'.format(save_name))
