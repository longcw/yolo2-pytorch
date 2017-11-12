import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, same_padding=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in list(net.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in list(net.state_dict().items()):
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def load_pretrained_npy(faster_rcnn_model, fname):
    params = np.load(fname).item()
    # vgg16
    vgg16_dict = faster_rcnn_model.rpn.features.state_dict()
    for name, val in list(vgg16_dict.items()):
        # # print name
        # # print val.size()
        # # print param.size()
        if name.find('bn.') >= 0:
            continue
        i, j = int(name[4]), int(name[6]) + 1
        ptype = 'weights' if name[-1] == 't' else 'biases'
        key = 'conv{}_{}'.format(i, j)
        param = torch.from_numpy(params[key][ptype])

        if ptype == 'weights':
            param = param.permute(3, 2, 0, 1)

        val.copy_(param)

    # fc6 fc7
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7'}
    for k, v in list(pairs.items()):
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda()
    return v


def variable_to_np_tf(x):
    return x.data.cpu().numpy().transpose([0, 2, 3, 1])


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)
