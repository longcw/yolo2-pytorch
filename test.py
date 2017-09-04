import os
import cv2
import torch
import numpy as np
import pickle

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.eval import class_AP
from datasets.lisa_hd import LISADataset
from datasets.egohands import EgoHandDataset
import cfgs.config as cfg


def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(
        yolo_utils.preprocess_test(image, cfg.inp_size), 0)
    return image, im_data


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
# trained_model = cfg.trained_model
# trained_model = os.path.join(cfg.train_output_dir, 'darknet19_egohands_debug_6.h5')
trained_model = 'models/training/darknet19_egohands_exp1/darknet19_egohands_exp1_5.h5'
trained_model = 'models/training/darknet19_lisa_exp1/darknet19_lisa_exp1_20.h5'
output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.4
vis = False


def test_net(net, imdb, max_per_image=300, thresh=0.5, vis=False):
    num_images = imdb.num_images

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    for image_idx in range(num_images):

        batch = imdb.next_batch()
        ori_im = batch['origin_im'][0]
        im_data = net_utils.np_to_variable(
            batch['images'], is_cuda=True, volatile=True).permute(0, 3, 1, 2)

        _t['im_detect'].tic()
        bbox_pred, iou_pred, prob_pred = net(im_data)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(
            bbox_pred, iou_pred, prob_pred, ori_im.shape, cfg, thresh)
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
                                      for class_idx in range(imdb.num_classes)])

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
                  {:.3f}s'.format(image_idx + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

        if vis:
            im2show = yolo_utils.draw_detection(ori_im, bboxes, scores,
                                                cls_inds, cfg, thr=0.1)
            if im2show.shape[0] > 1100:
                final_size = (int(1000. * float(im2show.shape[1]) /
                                  im2show.shape[0]), 1000)
                im2show = cv2.resize(im2show, final_size)
            cv2.imshow('test', im2show)
            cv2.waitKey(0)
    precision, recall = class_AP(imdb, all_boxes, class_name='hand', iou_thres=0.1)
    import pdb; pdb.set_trace()


    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    # data loader
    if cfg.dataset_name == 'lisa':
        imdb = LISADataset('train', cfg.DATA_DIR, cfg.batch_size,
                           yolo_utils.preprocess_test, processes=2,
                           shuffle=False, dst_size=cfg.inp_size,
                           use_cache=False)
    elif cfg.dataset_name == 'egohands':
        imdb = EgoHandDataset('test', cfg.DATA_DIR, cfg.batch_size,
                              yolo_utils.preprocess_test, processes=2,
                              shuffle=False, dst_size=cfg.inp_size,
                              differentiate_left_right=False,
                              use_cache=False)
    else:
        raise ValueError('dataset name {} \
                         not recognized'.format(cfg.dataset_name))
    print('load data succ...')

    net = Darknet19()
    net_utils.load_net(trained_model, net)

    net.cuda()
    net.eval()

    test_net(net, imdb, max_per_image, thresh, vis)

    imdb.close()
