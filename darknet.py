import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.network as net_utils
import cfgs.config as cfg
from layers.reorg.reorg_layer import ReorgLayer
from utils.cython_bbox import bbox_ious, bbox_intersections, bbox_overlaps
from utils.cython_yolo import yolo_to_bbox


def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels, ksize = item
                layers.append(net_utils.Conv2d_BatchNorm(in_channels, out_channels, ksize, same_padding=True))
                # layers.append(net_utils.Conv2d(in_channels, out_channels, ksize, same_padding=True))
                in_channels = out_channels

    return nn.Sequential(*layers), in_channels


class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()

        net_cfgs = [
            # conv1s
            [(32, 3)],
            ['M', (64, 3)],
            ['M', (128, 3), (64, 1), (128, 3)],
            ['M', (256, 3), (128, 1), (256, 3)],
            ['M', (512, 3), (256, 1), (512, 3), (256, 1), (512, 3)],
            # conv2
            ['M', (1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)],
            # ------------
            # conv3
            [(1024, 3), (1024, 3)],
            # conv4
            [(1024, 3)]
        ]

        # darknet
        self.conv1s, c1 = _make_layers(3, net_cfgs[0:5])
        self.conv2, c2 = _make_layers(c1, net_cfgs[5])
        # ---
        self.conv3, c3 = _make_layers(c2, net_cfgs[6])

        stride = 2
        self.reorg = ReorgLayer(stride=2)   # stride*stride times the channels of conv1s
        # cat [conv1s, conv3]
        self.conv4, c4 = _make_layers((c1*(stride*stride) + c3), net_cfgs[7])

        # linear
        out_channels = cfg.num_anchors * (cfg.num_classes + 5)
        self.conv5 = net_utils.Conv2d(c4, out_channels, 1, 1, relu=False)

        # train
        self.bbox_loss = None
        self.iou_loss = None
        self.cls_loss = None

    @property
    def loss(self):
        return self.bbox_loss * 5 + self.iou_loss + self.cls_loss

    def forward(self, im_data, gt_boxes=None, gt_classes=None, dontcare=None):
        conv1s = self.conv1s(im_data)

        conv2 = self.conv2(conv1s)

        conv3 = self.conv3(conv2)

        conv1s_reorg = self.reorg(conv1s)
        cat_1_3 = torch.cat([conv1s_reorg, conv3], 1)

        conv4 = self.conv4(cat_1_3)
        conv5 = self.conv5(conv4)   # batch_size, out_channels, h, w

        # for detection
        # bsize, c, h, w -> bsize, h, w, c -> bsize, h x w, num_anchors, 5+num_classes
        bsize, _, h, w = conv5.size()
        # assert bsize == 1, 'detection only support one image per batch'
        conv5_reshaped = conv5.permute(0, 2, 3, 1).contiguous().view(bsize, -1, cfg.num_anchors, cfg.num_classes + 5)

        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        xy_pred = F.sigmoid(conv5_reshaped[:, :, :, 0:2])
        wh_pred = torch.exp(conv5_reshaped[:, :, :, 2:4])
        bbox_pred = torch.cat([xy_pred, wh_pred], 3)

        iou_pred = F.sigmoid(conv5_reshaped[:, :, :, 4:5])

        score_pred = conv5_reshaped[:, :, :, 5:].contiguous()
        prob_pred = F.softmax(score_pred.view(-1, score_pred.size()[-1])).view_as(score_pred)

        # for training
        if self.training:
            bbox_pred_np = bbox_pred.data.cpu().numpy()
            _boxes, _ious, _classes, _mask = self._build_target(bbox_pred_np, gt_boxes, gt_classes, dontcare)
            _boxes = net_utils.np_to_variable(_boxes)
            _ious = net_utils.np_to_variable(_ious)
            _classes = net_utils.np_to_variable(_classes)
            _mask = net_utils.np_to_variable(_mask, dtype=torch.FloatTensor)
            num_boxes = _mask.data.sum()

            bbox_mask = _mask.expand_as(_boxes)
            self.bbox_loss = F.smooth_l1_loss(bbox_mask * bbox_pred, bbox_mask * _boxes,
                                         size_average=False) / num_boxes

            iou_mask = _mask * (iou_pred.data.numel() / num_boxes)
            self.iou_loss = nn.L1Loss()(iou_pred * iou_mask, _ious * iou_mask)

            cls_mask = _mask.expand_as(score_pred)
            self.cls_loss = nn.L1Loss(size_average=False)(prob_pred * cls_mask, _classes * cls_mask) / num_boxes
            # cls_loss = F.cross_entropy(score_pred.view(-1, score_pred.size()[-1]), _classes.view(-1))

            # print prob_pred.size(), _classes.size(), _mask.size()
            # cls_loss = nn.MSELoss()(prob_pred * _mask, _classes * _mask)
            # print num_boxes
            # print bbox_loss, iou_loss, cls_loss

        return bbox_pred, iou_pred, prob_pred

    def _build_target(self, bbox_pred_np, gt_boxes, gt_classes, dontcare):
        """
        :param bbox_pred: shape: (bsize, h x w, num_anchors, 4) : (sig(tx), sig(ty), exp(tw), exp(th))
        """
        W, H = cfg.out_size
        inp_size = cfg.inp_size
        out_size = cfg.out_size
        # TODO: dontcare areas
        # dontcare_areas = np.asarray(dontcare_areas, dtype=np.float)

        # net output
        bsize, hw, num_anchors, _ = bbox_pred_np.shape
        # gt
        _boxes = np.zeros([bsize, hw, num_anchors, 4], dtype=np.float)
        _ious = np.zeros([bsize, hw, num_anchors, 1], dtype=np.float)
        _classes = np.zeros([bsize, hw, num_anchors, cfg.num_classes], dtype=np.int)
        _mask = np.zeros([bsize, hw, num_anchors, 1], dtype=np.int)

        # scale pred_bbox
        anchors = np.ascontiguousarray(cfg.anchors, dtype=np.float)
        bbox_np = yolo_to_bbox(
            np.ascontiguousarray(bbox_pred_np, dtype=np.float),
            anchors,
            H, W)
        bbox_np[:, :, :, 0::2] *= float(inp_size[0])
        bbox_np[:, :, :, 1::2] *= float(inp_size[1])

        # assign each box to cells
        for b in range(bsize):
            gt_boxes_b = np.asarray(gt_boxes[b], dtype=np.float)

            # locate the cell of each gt_boxe
            cell_w = float(inp_size[0]) / W
            cell_h = float(inp_size[1]) / H
            cx = (gt_boxes_b[:, 0] + gt_boxes_b[:, 2]) * 0.5 / cell_w
            cy = (gt_boxes_b[:, 1] + gt_boxes_b[:, 3]) * 0.5 / cell_h
            cell_inds = np.floor(cy) * W + np.floor(cx)
            cell_inds = cell_inds.astype(np.int)
            # gt_boxes[:, :, 0::2] /= inp_size[1]
            # gt_boxes[:, :, 1::2] /= inp_size[0]

            target_boxes = np.empty(gt_boxes_b.shape, dtype=np.float)
            target_boxes[:, 0] = cx - np.floor(cx)  # cx
            target_boxes[:, 1] = cy - np.floor(cy)  # cy
            target_boxes[:, 2] = (gt_boxes_b[:, 2] - gt_boxes_b[:, 0]) / inp_size[0] * out_size[0]  # tw
            target_boxes[:, 3] = (gt_boxes_b[:, 3] - gt_boxes_b[:, 1]) / inp_size[1] * out_size[1]  # th

            cell_boxes = [[] for _ in range(hw)]
            for i, ind in enumerate(cell_inds):
                # print ind
                if ind >= hw or ind < 0:
                    print ind
                    continue
                cell_boxes[ind].append(i)

            for i in range(hw):
                if len(cell_boxes[i]) == 0:
                    continue
                bboxes = [gt_boxes_b[j] for j in cell_boxes[i]]
                targets_b = np.array([target_boxes[j] for j in cell_boxes[i]], dtype=np.float)
                targets_c = np.array([gt_classes[b][j] for j in cell_boxes[i]], dtype=np.int)

                ious = bbox_ious(
                    np.ascontiguousarray(bbox_np[b, i], dtype=np.float),
                    np.ascontiguousarray(bboxes, dtype=np.float)
                )

                argmax = np.argmax(ious, axis=0)
                for j, a in enumerate(argmax):
                    if _ious[b, i, a, 0] <= ious[a, j]:
                        _mask[b, i, a, :] = 1
                        _ious[b, i, a, 0] = ious[a, j]
                        targets_b[j, 2:4] /= anchors[a]
                        _boxes[b, i, a, :] = targets_b[j]
                        # print bbox_pred_np[b, i, a], targets_b[j]
                        # _classes[b, i, a, :] = targets_c[j]
                        _classes[b, i, a, targets_c[j]] = 1

            # _boxes[:, :, :, 2:4] /= anchors

                #
                # _boxes[b, i, :, :] = _box
                # _ious[b, i, :, :] = np.expand_dims(ious[(np.arange(len(argmax)), argmax)], 1)
                # _classes[b, i, :, targets_c[argmax]] = 1
                #
                # _mask[b, i, :, :] = 1
        return _boxes, _ious, _classes, _mask

    def load_from_npz(self, fname, num_conv=None):
        dest_src = {'conv.weight': 'kernel', 'conv.bias': 'biases',
                    'bn.weight': 'gamma', 'bn.bias': 'biases',
                    'bn.running_mean': 'moving_mean', 'bn.running_var': 'moving_variance'}
        params = np.load(fname)
        own_dict = self.state_dict()
        keys = own_dict.keys()

        for i, start in enumerate(range(0, len(keys), 5)):
            if num_conv is not None and i>= num_conv:
                break
            end = min(start+5, len(keys))
            for key in keys[start:end]:
                list_key = key.split('.')
                ptype = dest_src['{}.{}'.format(list_key[-2], list_key[-1])]
                src_key = '{}-convolutional/{}:0'.format(i, ptype)
                print(src_key, own_dict[key].size(), params[src_key].shape)
                param = torch.from_numpy(params[src_key])
                if ptype == 'kernel':
                    param = param.permute(3, 2, 0, 1)
                own_dict[key].copy_(param)

if __name__ == '__main__':
    net = Darknet19()
    # net.load_from_npz('models/yolo-voc.weights.npz')
    net.load_from_npz('models/darknet19.weights.npz', num_conv=18)

