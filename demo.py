import torch
import numpy as np

from darknet import Darknet19

net = Darknet19()
net.load_from_npz('models/yolo-voc.weights.npz')