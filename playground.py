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
def skiToTorch(ski_image):
    transposed = ski_image.transpose((2, 0, 1))
    return torch.from_numpy(transposed)
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['tags']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
class LISADataset(Dataset):
    """LISA Traffic Sign Dataset dataset."""

    def __init__(self, image_names_csv_file, tags_csv_file, root_dir, transform=None):
        """
        Args:
            image_names_csv_file (string): Path to the csv file with image paths.
            tags_csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(image_names_csv_file, delimiter=';')
        self.tags = pd.read_csv(tags_csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.category_name_to_id = {
                                'top',1
                                'speedLimitUrdbl',2
                                'speedLimit25',3
                                'pedestrianCrossing',4
                                'speedLimit35',5
                                'turnLeft',6
                                'slow',7
                                'speedLimit15',8
                                'speedLimit45',9
                                'rightLaneMustTurn',10
                                'signalAhead',11
                                'keepRight',12
                                'laneEnds',13
                                'school',14
                                'merge',15
                                'addedLane',16
                                'rampSpeedAdvisory40',17
                                'rampSpeedAdvisory45',18
                                'curveRight',19
                                'speedLimit65',20
                                'truckSpeedLimit55',21
                                'thruMergeLeft',22
                                'speedLimit30',23
                                'stopAhead',24
                                'yield',25
                                'thruMergeRight',26
                                'dip',27
                                'schoolSpeedLimit25',28
                                'thruTrafficMergeLeft',29
                                'noRightTurn',30
                                'rampSpeedAdvisory35',31
                                'curveLeft',32
                                'rampSpeedAdvisory20',33
                                'noLeftTurn',34
                                'zoneAhead25',35
                                'zoneAhead45',36
                                'doNotEnter',37
                                'yieldAhead',38
                                r'oundabout',39
                                'turnRight',40
                                'speedLimit50',41
                                'rampSpeedAdvisoryUrdbl',42
                                'rampSpeedAdvisory50',43
                                'speedLimit40',44
                                'speedLimit55',45
                                'doNotPass',46
                                'intersection',47
        }

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images.ix[idx, 0])
        image = cv2.imread(img_name)
        image = im_transform.imcv2_recolor(image)
        image_class = self.tags.ix[idx, 0]
        #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y
        vals = self.tags.ix[idx, 1:].as_matrix().astype('int')
        #Darknet wants x_bottom_left, y_bottom_left, x_top_right, and y_top_right
        tags = numpy.zeros(4)
        tags[0]=vals[
        tags[1]=width
        tags[2]=width
        tags[3]=width
        sample = {'image': image, 'class': image_class, 'tags': tags}

        if self.transform:
            sample = self.transform(sample)

        return sample
images = pd.read_csv('datasets/LISA/allAnnotations.csv', delimiter=';')
tags = pd.read_csv('datasets/LISA/tags.csv')
ds = LISADataset('datasets/LISA/allAnnotations.csv', 'datasets/LISA/tags.csv', 'datasets/LISA')
print(ds.__getitem__(10))