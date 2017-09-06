import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

import cfgs.config as cfg
from darknet import Darknet19
from datasets.gteagazeplusimage import GTEAGazePlusImage
from datasets.utils.visualize import plot_bboxes
import utils.network as net_utils
import utils.yolo as yolo_utils



def test_net(net, dataset, transform=None, max_per_image=300, thresh=0.5,
             num_classes=1, vis=False):

    # Run through dataset
    for i, (img, annots) in tqdm(enumerate(dataset)):
        original_img = np.array(img)
        if transform is not None:
            img = transform(img)

            # Add batch dimension
            img = img.unsqueeze(0)

            # Create GPU variable
            img_var = Variable(img.type(torch.FloatTensor))
            img_var = img_var.cuda()

            # Detect hands
            bbox_pred, iou_pred, prob_pred = net(img_var)

            # to numpy
            bbox_pred = bbox_pred.data.cpu().numpy()
            iou_pred = iou_pred.data.cpu().numpy()
            prob_pred = prob_pred.data.cpu().numpy()

            bboxes, scores, cls_inds = yolo_utils.postprocess(
                bbox_pred, iou_pred, prob_pred,
                original_img.shape[0:2], cfg, thresh)

            for class_idx in range(num_classes):
                # Extract class
                inds = np.where(cls_inds == class_idx)[0]

                class_bboxes = bboxes[inds]
                class_scores = scores[inds]
                class_scores = class_scores[:, np.newaxis]

                # Create class detections in format
                # [[x_min, y_min, x_max, y_max, score], ...]
                if vis:
                    plot_bboxes(original_img, class_bboxes)


if __name__ == "__main__":
    vis = True

    # Initialize dataset
    dataset = GTEAGazePlusImage()

    # Initialize test image transform
    test_transform = transforms.Compose([
        transforms.Scale(cfg.inp_size),
        transforms.ToTensor()])

    # Initialise network
    trained_model = 'models/training/darknet19_all_exp1/darknet19_all_exp1_2.h5'
    net = Darknet19()
    net_utils.load_net(trained_model, net)
    net.cuda()
    net.eval()


    # Extract bounding boxes
    test_net(net, dataset, transform=test_transform, vis=vis)
