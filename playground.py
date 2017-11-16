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
from datasets.lisa_dataset import LISADataset


images_file = 'datasets/LISA/allAnnotations.csv'
tags_file = 'datasets/LISA/tags.csv'
root_dir = 'datasets/LISA'
ds = LISADataset(images_file, tags_file, root_dir)
images, labels, classes = ds.getBatch(0, batch_size = 8)
print(images)
print(labels[0].shape)
print(classes[0].shape)
 
