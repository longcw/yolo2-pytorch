import os
import pickle

import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.eval import class_AP
from datasets.lisa_hd import LISADataset
from datasets.egohands import EgoHandDataset
from datasets.synthetic import SyntheticDataset
from datasets.pascal_voc import VOCDataset
import cfgs.config as cfg

from datasets.voc_eval import voc_ap, voc_eval
from datasets.utils.augmentation import data_augmentation
from matplotlib import pyplot as plt


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
print('dataset name', imdb_name)
# trained_model = cfg.trained_model
# trained_model = os.path.join(cfg.train_output_dir, 'darknet19_egohands_debug_6.h5')
# trained_model = 'models/yolo-voc.weights.h5'
# trained_model = 'models/training/darknet19_lisa_exp1/darknet19_lisa_exp1_20.h5'
# trained_model = 'models/training/darknet19_egohands_exp1/darknet19_egohands_exp1_5.h5'
trained_model = 'models/training/darknet19_all_exp1/darknet19_all_exp1_15.h5'

output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.01
vis = True


def test_net(net, dataloader, max_per_image=300, thresh=0.5,
             vis=False, use_cache=False):

    num_images = len(dataloader.dataset)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    if use_cache:
        with open(det_file, 'rb') as f:
            all_boxes = pickle.load(f)
    else:
        for image_idx, (img, original_img, targets) in enumerate(dataloader):
            img_var = Variable(img.type(torch.FloatTensor))
            img_var = img_var.cuda()

            # Convert PIL image to cv2 numpy array
            original_img = original_img[0].numpy()
            original_img = original_img[:, :, ::-1].copy()

            _t['im_detect'].tic()
            bbox_pred, iou_pred, prob_pred = net(img_var)

            # to numpy
            bbox_pred = bbox_pred.data.cpu().numpy()
            iou_pred = iou_pred.data.cpu().numpy()
            prob_pred = prob_pred.data.cpu().numpy()

            bboxes, scores, cls_inds = yolo_utils.postprocess(
                bbox_pred, iou_pred, prob_pred, original_img.shape[0:2], cfg, thresh)
            detect_time = _t['im_detect'].toc()

            _t['misc'].tic()

            for class_idx in range(imdb.num_classes):
                # Extract class
                inds = np.where(cls_inds == class_idx)[0]
                if len(inds) == 0:
                    all_boxes[class_idx][image_idx] = np.empty([0, 5],
                                                               dtype=np.float32)
                    continue
                class_bboxes = bboxes[inds]
                class_scores = scores[inds]
                class_scores = class_scores[:, np.newaxis]

                # Create class detections in format
                # [[x_min, y_min, x_max, y_max, score], ...]
                class_detections = np.hstack((class_bboxes,
                                              class_scores)).astype(np.float32,
                                                                    copy=False)
                all_boxes[class_idx][image_idx] = class_detections

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[class_idx][image_idx][:, -1]
                                          for class_idx
                                          in range(dataloader.dataset.num_classes)])

                # Only keep max_per_image detections with highest scores
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for class_idx in range(1, imdb.num_classes):
                        keep = np.where(
                            all_boxes[class_idx][image_idx][:, -1] >= image_thresh)[0]
                        all_boxes[class_idx][image_idx] = all_boxes[class_idx][image_idx][keep, :]
            nms_time = _t['misc'].toc()

            if image_idx % 20 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s \
                      {:.3f}s'.format(image_idx + 1, num_images,
                                      detect_time, nms_time))
                _t['im_detect'].clear()
                _t['misc'].clear()

            if vis:
                im2show = yolo_utils.draw_detection(original_img, bboxes,
                                                    scores, cls_inds,
                                                    cfg, thr=0.5)
                if im2show.shape[0] > 1100:
                    final_size = (int(1000. * float(im2show.shape[1]) /
                                      im2show.shape[0]), 1000)
                    im2show = cv2.resize(im2show, final_size)
                cv2.imshow('test', im2show)
                cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    class_name = 'hand'

    # Compute class Average Precision
    precision, recall = class_AP(imdb, all_boxes,
                                 class_name=class_name, iou_thres=0.5)
    ap = voc_ap(recall, precision, use_07_metric=False)
    print('ap {} for class {}'.format(ap, class_name))


if __name__ == '__main__':

    # Create test transforms
    test_transform = transforms.Compose([
        transforms.Scale(cfg.inp_size),
        transforms.ToTensor()])

    # Initialize dataset
    if cfg.dataset_name == 'lisa':
        imdb = LISADataset('train', cfg.DATA_DIR, transform=test_transform,
                           use_cache=False)
    elif cfg.dataset_name == 'egohands':
        imdb = EgoHandDataset('test', cfg.DATA_DIR, transform=test_transform,
                              use_cache=False)
    elif cfg.dataset_name == 'synthetic':
        imdb = SyntheticDataset(cfg.DATA_DIR, transform=test_transform,
                                use_cache=False)
    elif cfg.dataset_name == 'voc':
        imdb = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                          yolo_utils.preprocess_test, processes=2,
                          shuffle=False, dst_size=cfg.inp_size,
                          use_07_metric=False, use_cache=False)

    else:
        raise ValueError('dataset name {} \
                         not recognized'.format(cfg.dataset_name))
    print('load data succ...')

    # Initialize dataloader
    dataloader = DataLoader(imdb, shuffle=False, batch_size=cfg.batch_size,
                            num_workers=2)

    # Initialize network
    net = Darknet19()
    net_utils.load_net(trained_model, net)

    net.cuda()
    net.eval()

    test_net(net, dataloader, max_per_image, thresh, vis, use_cache=False)
