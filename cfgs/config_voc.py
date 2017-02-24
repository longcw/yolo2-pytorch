import numpy as np


inp_size = np.array([416, 416], dtype=np.int)
out_size = inp_size / 32

# [region]
label_names = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
classes = len(label_names)
anchors = [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)]


def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127

base = int(np.ceil(pow(classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(classes)]


bias_match = 1

coords = 4

softmax = 1
jitter = .2
rescore = 1

object_scale = 5
noobject_scale = 1
class_scale = 1
coord_scale = 1

absolute = 1
thresh = .3
random = 0
