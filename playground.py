from __future__ import print_function, division
import os
import csv
from skimage import io, transform
import pandas as pd
import torch
import numpy as np
import cv2
from utils import im_transform
from datasets.imdb import ImageDataset
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
        image, img_class, tags = sample['image'], sample['class'], sample['tags']

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
        dx = float(new_w) / float(w)
        dy = float(new_h) / float(h)
        tags[0] = int(float(tags[0])*dx)
        tags[2] = int(float(tags[2])*dx)
        tags[1] = int(float(tags[1])*dy)
        tags[3] = int(float(tags[3])*dy)

        return {'image': img, 
                'class': img_class,
                'tags': tags}
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, img_class, tags = sample['image'], sample['class'], sample['tags']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'class': img_class,
                'tags': torch.from_numpy(tags).int()}
class LISADataset():
    """LISA Traffic Sign Dataset dataset."""

    def __init__(self, image_names_csv_file, tags_csv_file, root_dir):
        """
        Args:
            image_names_csv_file (string): Path to the csv file with image paths.
            tags_csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_file = pd.read_csv(image_names_csv_file, delimiter=';')
        self.tags = pd.read_csv(tags_csv_file)
        self.root_dir = root_dir
        self.category_name_to_id = {'stop':1,
                                    'speedLimitUrdbl':2,
                                    'speedLimit25':3,
                                    'pedestrianCrossing':4,
                                    'speedLimit35':5,
                                    'turnLeft':6,
                                    'slow':7,
                                    'speedLimit15':8,
                                    'speedLimit45':9,
                                    'rightLaneMustTurn':10,
                                    'signalAhead':11,
                                    'keepRight':12,
                                    'laneEnds':13,
                                    'school':14,
                                    'merge':15,
                                    'addedLane':16,
                                    'rampSpeedAdvisory40':17,
                                    'rampSpeedAdvisory45':18,
                                    'curveRight':19,
                                    'speedLimit65':20,
                                    'truckSpeedLimit55':21,
                                    'thruMergeLeft':22,
                                    'speedLimit30':23,
                                    'stopAhead':24,
                                    'yield':25,
                                    'thruMergeRight':26,
                                    'dip':27,
                                    'schoolSpeedLimit25':28,
                                    'thruTrafficMergeLeft':29,
                                    'noRightTurn':30,
                                    'rampSpeedAdvisory35':31,
                                    'curveLeft':32,
                                    'rampSpeedAdvisory20':33,
                                    'noLeftTurn':34,
                                    'zoneAhead25':35,
                                    'zoneAhead45':36,
                                    'doNotEnter':37,
                                    'yieldAhead':38,
                                    'roundabout':39,
                                    'turnRight':40,
                                    'speedLimit50':41,
                                    'rampSpeedAdvisoryUrdbl':42,
                                    'rampSpeedAdvisory50':43,
                                    'speedLimit40':44,
                                    'speedLimit55':45,
                                    'doNotPass':46,
                                    'intersection':47
                                   }
        self.ix = 0
    def toTensor(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
    def rescaleTags(self, tags, dx, dy):
        tags[0] = int(float(tags[0])*dx)
        tags[2] = int(float(tags[2])*dx)
        tags[1] = int(float(tags[1])*dy)
        tags[3] = int(float(tags[3])*dy)
        return tags
    def rescaleImage(self, image, output_size):

        h, w = image.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        dx = float(new_w) / float(w)
        dy = float(new_h) / float(h)
        return image, dx, dy
    def fileToTensor(self, file):
        image = cv2.imread(file)
        image = im_transform.imcv2_recolor(image)
        image, dx, dy = self.rescaleImage(image, (416,416))
        image_tensor = self.toTensor(image).float()
        return image_tensor, dx, dy
    def nextBatch(self, batch_size = 8):
        current_img_file = self.images_file.ix[self.ix, 0]
        fp = os.path.join(self.root_dir, current_img_file)
        print('loading: ' + fp)
        tensor, dx, dy = self.fileToTensor(fp)
        images = torch.FloatTensor(1,3,416,416)
        images[0] = tensor.clone()
        image_class = self.tags.ix[self.ix, 0]
        class_id = self.category_name_to_id[image_class]
        #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y
        vals = self.tags.ix[self.ix, 1:].as_matrix().astype('int')
        #Darknet wants x_bottom_left, y_bottom_left, x_top_right, and y_top_right
        tags = np.zeros(4)
        tags[0]=vals[0]
        tags[2]=vals[2]
        tags[1]=vals[3]
        tags[3]=vals[1]
        tags = self.rescaleTags(tags,dx,dy)
        classes = []
        labels = []
        labels.append(np.array([tags]))
        classes.append(np.array([class_id]))
        batch_index = 0
        self.ix = self.ix + 1
        while(True):
            
            img_file =  self.images_file.ix[self.ix, 0]
            if(img_file == current_img_file):
                print('file is the same')
                image_class = self.tags.ix[self.ix, 0]
                class_id = self.category_name_to_id[image_class]
                #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y
                vals = self.tags.ix[self.ix, 1:].as_matrix().astype('int')
                #Darknet wants x_bottom_left, y_bottom_left, x_top_right, and y_top_right
                tags = np.zeros(4)
                tags[0]=vals[0]
                tags[2]=vals[2]
                tags[1]=vals[3]
                tags[3]=vals[1]
                tags = self.rescaleTags(tags,dx,dy)
                np.concatenate((labels[-1], [tags]))
                np.concatenate((classes[-1], np.array([class_id])))
                print(classes)
                print(labels)
            else:
                batch_index = batch_index + 1
                if(batch_index>=batch_size):
                    break
                current_img_file = img_file
                fp = os.path.join(self.root_dir, current_img_file)
                print('loading: ' + fp)
                tensor, dx, dy = self.fileToTensor(fp)
                temp = torch.FloatTensor(1,3,416,416)
                temp[0]=tensor
                images = torch.cat((images,temp))
                image_class = self.tags.ix[self.ix, 0]
                class_id = self.category_name_to_id[image_class]
                #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y
                vals = self.tags.ix[self.ix, 1:].as_matrix().astype('int')
                #Darknet wants x_bottom_left, y_bottom_left, x_top_right, and y_top_right
                tags = np.zeros(4)
                tags[0]=vals[0]
                tags[2]=vals[2]
                tags[1]=vals[3]
                tags[3]=vals[1]
                tags = self.rescaleTags(tags,dx,dy)
                labels.append(np.array([tags]))
                classes.append(np.array([class_id]))
            self.ix = self.ix + 1
        return images, labels, classes     
                
            
               
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images_file.ix[idx, 0])
        image = cv2.imread(img_name)
        image = im_transform.imcv2_recolor(image)
        image_class = self.tags.ix[idx, 0]
        #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y

        vals = self.tags.ix[idx, 1:].as_matrix().astype('int')
        #Darknet wants x_bottom_left, y_bottom_left, x_top_right, and y_top_right
        tags = np.zeros(4)
        tags[0]=vals[0]
        tags[2]=vals[2]
        tags[1]=vals[3]
        tags[3]=vals[1]
        sample = {'image': image, 'class': self.category_name_to_id[image_class], 'tags': tags}
        if self.transform:
            sample = self.transform(sample)

        return sample


images_file = 'datasets/LISA/allAnnotations.csv'
tags_file = 'datasets/LISA/tags.csv'
root_dir = 'datasets/LISA'
ds = LISADataset(images_file, tags_file, root_dir)
images, labels, classes = ds.nextBatch(batch_size = 4)
print(images)
print(labels)
print(classes)

