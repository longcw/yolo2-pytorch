import os
from config_voc import *
from exps.darknet19_exp1 import *


def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


# input and output size
############################
inp_size = np.array([416, 416], dtype=np.int)
out_size = inp_size / 32


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127

base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]


# detection config
############################
thresh = 0.3


# train config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'data'))

pretrained_model = os.path.join('models', 'darknet19.weights.npz')
output_dir = os.path.join('models', 'training', exp_name)
mkdir(output_dir, max_depth=3)

rand_seed = 1024
use_tensorboard = True

max_epoch = 10
max_step = 1000
