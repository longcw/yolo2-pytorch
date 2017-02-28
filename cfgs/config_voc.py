import numpy as np


# VOC
label_names = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
num_classes = len(label_names)

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)
num_anchors = len(anchors)

# dataset
imdb_name = 'trainval'
year = 2007
batch_size = 1


#
# bias_match = 1
#
# coords = 4
#
# softmax = 1
# jitter = .2
# rescore = 1
#
# object_scale = 5
# noobject_scale = 1
# class_scale = 1
# coord_scale = 1
#
# absolute = 1

# random = 0
