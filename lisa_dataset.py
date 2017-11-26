import os
import cv2
import torch
import numpy as np
import datetime
from torch.multiprocessing import Pool
import pandas as pd
from darknet import Darknet19
from datasets.pascal_voc import VOCDataset
from utils import im_transform
from skimage import io, transform
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
try:
   # 
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

def imcv2_recolor(im, a=.1):
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im
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
        self.current_ix = 0
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
        return tags.astype(int)
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
        image = imcv2_recolor(image)
        image, dx, dy = self.rescaleImage(image, (416,416))
        image_tensor = self.toTensor(image).float()
        return image_tensor, dx, dy
    def loadOnlyTags(self, ix, dx, dy):
        image_class = self.tags.ix[ix, 0]
        class_id = self.category_name_to_id[image_class]
        #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y
        vals = self.tags.ix[ix, 1:].as_matrix().astype(int)
        #Darknet wants x_bottom_left, y_bottom_left, x_top_right, and y_top_right

        tags=vals
        tags = self.rescaleTags(tags,dx,dy)
        return tags, class_id
    def dataPoint(self, ix, load_image=True): 
        local_fp = self.images_file.ix[ix, 0]
        fp = os.path.join(self.root_dir, local_fp)
        print('loading: ' + fp)
        tensor, dx, dy = self.fileToTensor(fp)
        image_class = self.tags.ix[ix, 0]
        class_id = self.category_name_to_id[image_class]
        #Annotation tag: Upper left corner X, Upper left corner Y, Lower right corner X, Lower right corner Y
        vals = self.tags.ix[ix, 1:].as_matrix().astype(int)

        tags=vals
        tags = self.rescaleTags(tags,dx,dy)       
        return local_fp, tensor, dx, dy, tags, class_id
    def reset(self):
        self.current_ix=0
    def hasMoreImages(self):
        return self.current_ix<self.images_file.shape[0]
    def getImage(self):
        if(self.current_ix>=self.images_file.shape[0]):
            raise IndexError('no more')
        local_fp, tensor, dx, dy, locations, class_id = self.dataPoint(self.current_ix)
        current_img_file = local_fp
        locations = np.array([locations])
        class_ids = np.array([class_id])
        while(True):
            self.current_ix = self.current_ix + 1
            if(self.current_ix>=self.images_file.shape[0]):
                break
            img_file =  self.images_file.ix[self.current_ix, 0]
            if(img_file == current_img_file):
         #       print('file is the same')
                tags, class_id = self.loadOnlyTags(self.current_ix, dx, dy)
                locations = np.concatenate((locations, np.array([tags])))
                class_ids = np.concatenate((class_ids, np.array([class_id])))
            else:
                break
        return tensor, locations, class_ids
    def getBatch(self, batch_size = 8):
        images = torch.FloatTensor(1,3,416,416)
        temp = torch.FloatTensor(1,3,416,416)
        labels = []
        classes = []
        for i in range(0, batch_size):
            tensor, locations, class_ids = self.getImage()
            if(i == 0):
                images[0] = tensor.clone()
            else:
                temp[0]=tensor.clone()
                images = torch.cat((images,temp))
            labels.append(locations)
            classes.append(class_ids)
        return images, labels, classes