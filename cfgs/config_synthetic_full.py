import os

import numpy as np

# # Exp params
exp_name = 'darknet19_synthetic_exp1'

pretrained_fname = 'darknet19.weights.npz'

start_step = 0
lr_decay_epochs = {60, 90}
lr_decay = 1./10

max_epoch = 160

weight_decay = 0.0005
momentum = 0.9
init_learning_rate = 1e-3

# for training yolo2
object_scale = 5.
noobject_scale = 1.
class_scale = 1.
coord_scale = 1.
iou_thresh = 0.6

# # Dataset params
# trained model
h5_fname = 'yolo-synthetic.weights.h5'

# Classes
differentiate_left_right = False
label_names = (['hand'])
num_classes = len(label_names)

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)
num_anchors = len(anchors)

# dataset
imdb_train = 'synthetic'
imdb_test = 'synthetic'
batch_size = 1

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


inp_size = np.array([416, 416], dtype=np.int)   # w, h
out_size = inp_size / 32


# Display params
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


# dir config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TRAIN_DIR = os.path.join(MODEL_DIR, 'training')
TEST_DIR = os.path.join(MODEL_DIR, 'testing')

trained_model = os.path.join(MODEL_DIR, h5_fname)
pretrained_model = os.path.join(MODEL_DIR, pretrained_fname)

train_output_dir = os.path.join(TRAIN_DIR, exp_name)
test_output_dir = os.path.join(TEST_DIR, imdb_test, h5_fname)
mkdir(train_output_dir, max_depth=3)
mkdir(test_output_dir, max_depth=4)

rand_seed = 1024
train_batch_size = 16
use_tensorboard = True

log_interval = 50
disp_interval = 10
