import os
import cv2
import numpy as np
from torch.multiprocessing import Pool

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg

# This prevents deadlocks in the data loader, caused by
# some incompatibility between pytorch and cv2 multiprocessing.
# See https://github.com/pytorch/pytorch/issues/1355.
cv2.setNumThreads(0)


def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(
        yolo_utils.preprocess_test((image, None, cfg.multi_scale_inp_size), 0)[0], 0)
    return image, im_data


# hyper-parameters
# npz_fname = 'models/yolo-voc.weights.npz'
# h5_fname = 'models/yolo-voc.weights.h5'
trained_model = cfg.trained_model
# trained_model = os.path.join(
#     cfg.train_output_dir, 'darknet19_voc07trainval_exp3_158.h5')
thresh = 0.5
im_path = 'demo'
# ---

net = Darknet19()
net_utils.load_net(trained_model, net)
# net.load_from_npz(npz_fname)
# net_utils.save_net(h5_fname, net)
net.cuda()
net.eval()
print('load model succ...')

t_det = Timer()
t_total = Timer()
im_fnames = sorted((fname
                    for fname in os.listdir(im_path)
                    if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
pool = Pool(processes=1)

for i, (image, im_data) in enumerate(pool.imap(
        preprocess, im_fnames, chunksize=1)):
    t_total.tic()
    im_data = net_utils.np_to_variable(
        im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
    t_det.tic()
    bbox_pred, iou_pred, prob_pred = net(im_data)
    det_time = t_det.toc()
    # to numpy
    bbox_pred = bbox_pred.data.cpu().numpy()
    iou_pred = iou_pred.data.cpu().numpy()
    prob_pred = prob_pred.data.cpu().numpy()

    # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

    bboxes, scores, cls_inds = yolo_utils.postprocess(
        bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)

    im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. *
                                  float(im2show.shape[1]) / im2show.shape[0]),
                              1000))
    cv2.imshow('test', im2show)

    total_time = t_total.toc()
    # wait_time = max(int(60 - total_time * 1000), 1)
    cv2.waitKey(0)

    if i % 1 == 0:
        format_str = 'frame: %d, ' \
                     '(detection: %.1f Hz, %.1f ms) ' \
                     '(total: %.1f Hz, %.1f ms)'
        print((format_str % (
            i,
            1. / det_time, det_time * 1000,
            1. / total_time, total_time * 1000)))

        t_total.clear()
        t_det.clear()
