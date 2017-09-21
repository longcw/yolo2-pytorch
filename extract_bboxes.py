import numpy as np
import os
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

import cfgs.config as cfg
from darknet import Darknet19
from datasets.gteagazeplusimage import GTEAGazePlusImage
from datasets.smthgimage import SmthgImage
from datasets.utils.visualize import plot_bboxes
import utils.network as net_utils
import utils.yolo as yolo_utils


def get_crop_params(bbox, img_shape, increase_ratio=2.2):
    """
    returns x_min, y_min, x_max ,y_max crop coordinates according to rule
    2.2 times max dimension of the bounding box

    Args:
        bbox(numpy.ndarray): x_min, y_min, x_max, y_max
        img_shape(tuple): original image shape
        increase_ratio(float): final bbox size / tight bbox size
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    square_dim = max(width, height)
    final_dim = square_dim * increase_ratio
    center_x = bbox[0] + width / 2
    center_y = bbox[1] + height / 2
    new_x_min = int(center_x - final_dim / 2)
    new_x_max = int(center_x + final_dim / 2)
    new_y_min = int(center_y - final_dim / 2)
    new_y_max = int(center_y + final_dim / 2)
    if new_x_min >= 0 and new_y_min >= 0 and\
            new_x_max <= img_shape[0] and new_y_max <= img_shape[1]:
        success = True
    else:
        success = False

    return success, (new_x_min, new_y_min, new_x_max, new_y_max)


def test_net(net, dataset, transform=None, max_per_image=300, thresh=0.5,
             num_classes=1, vis=False, crop_folder=None):

    # Initialize counter for number of cropped hands
    extracted_hands = 0

    # Run through dataset
    for i, (img, annots) in tqdm(enumerate(dataset)):
        original_img = img
        np_original_img = np.array(original_img)
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
                np_original_img.shape[0:2], cfg, thresh)

            for class_idx in range(num_classes):
                # Extract class
                inds = np.where(cls_inds == class_idx)[0]

                class_bboxes = bboxes[inds]
                class_scores = scores[inds]
                class_scores = class_scores[:, np.newaxis]

                # Create class detections in format
                # [[x_min, y_min, x_max, y_max, score], ...]
                if vis:
                    fig = plot_bboxes(np_original_img, class_bboxes,
                                class_scores)
                    fig.savefig('bboxes_{:03}.jpg'.format(i), bbox_inches='tight')

                # Save crops to (368x368) images
                if crop_folder is not None:
                    for i, bbox in enumerate(class_bboxes):
                        crop_success, crop_params = get_crop_params(bbox,
                                                                    (original_img.width,
                                                                     original_img.height))
                        if crop_success:
                            crop = original_img.crop((crop_params))
                            crop_name = 'rendered_{:03d}.jpg'.format(
                                extracted_hands)
                            crop = crop.resize((368, 368))
                            # if bbox[2] - bbox[0] > 100 and bbox[3] - bbox[1] > 100:
                            # Mirror left hands
                            if cls_inds[i] == 0:
                                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                            print('saving image')
                            crop.save(os.path.join(crop_folder, crop_name))
                            extracted_hands += 1


if __name__ == "__main__":
    vis = True
    crop_folder = 'results/crops'

    # Initialize dataset
    dataset = GTEAGazePlusImage()
    # dataset = SmthgImage()

    # Initialize test image transform
    test_transform = transforms.Compose([
        transforms.Scale(cfg.inp_size),
        transforms.ToTensor()])

    # Initialise network
    # trained_model = 'models/training/darknet19_all_exp1/darknet19_all_exp1_64.h5'
    trained_model = 'models/training/darknet19_all_exp1/darknet19_all_exp1_15.h5'

    net = Darknet19()
    net_utils.load_net(trained_model, net)
    net.cuda()
    net.eval()

    # Extract bounding boxes
    test_net(net, dataset, transform=test_transform, vis=vis, thresh=0.5,
             crop_folder=None)
