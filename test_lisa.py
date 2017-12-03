import os
import cv2
import torch
import numpy as np
import cPickle

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer

from datasets.pascal_lisa import PascalLISADataset

import cfgs.config as cfg


def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(yolo_utils.preprocess_test(image, cfg.inp_size), 0)
    return image, im_data



output_dir = 'test_results'

max_per_image = 300
thresh = 0.5
vis = False
# ------------
cuda = True
det_file = os.path.join('test_results', 'detections.pkl')
def test_net(net, imdb, max_per_image=300, thresh=0.5, vis=False):
    num_images = 1569

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(47)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    #det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(0, num_images):
        print('testing on the ' + str(i+1) + 'th batch')
        batch = imdb.next_batch()
        ori_im = batch['origin_im'][0]
        gt_boxes = batch['gt_boxes']
        gt_classes = batch['gt_classes']
        im_data = net_utils.np_to_variable(batch['images'], is_cuda=cuda, volatile=True).permute(0, 3, 1, 2)
       # print("got im_data")
        _t['im_detect'].tic()
    ##    print(im_data.size())
        #print(im_data)
        bbox_pred, iou_pred, prob_pred = net(im_data)
        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()
        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, ori_im.shape, cfg, thresh)
        print('Predicted boxes: ', bboxes)
        print('Predicted classes: ', cls_inds)
        
        
        #print('Ground truth boxes: ', gt_boxes)
        #print('Ground truth classes: ', gt_classes)

        if vis:
            im2show = yolo_utils.draw_detection(ori_im, bboxes, scores, cls_inds, cfg, thr=0.1)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show, (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
            cv2.imshow('test', im2show)
            cv2.waitKey(0)
        else:
            im2show = yolo_utils.draw_detection(ori_im, bboxes, scores, cls_inds, cfg, thr=0.1)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show, (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))            
            output_path=os.path.join('output_images','test_image_'+str(i)+'.jpg')
            cv2.imwrite(output_path,im2show)

       


    
if __name__ == '__main__':
    # data loader
    imdb = PascalLISADataset("LISA","data", 1, yolo_utils.preprocess_test, processes=1, shuffle=True, dst_size=cfg.inp_size, val=True)
    print('load data succ...')

    net = Darknet19()
    net_utils.load_net(os.path.join('lisa_models', 'yolo_lisa_with_synthetic_data_99.h5'), net)
    print('load net succ...')
    if cuda:
        net.cuda()
    net.eval()

    test_net(net, imdb, max_per_image, thresh, vis)

    imdb.close()
