import random

import numpy as np
from PIL import Image


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if random.randint(1, 10000) % 2:
        return scale
    else:
        return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def image_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(0, dw)
    pright = random.randint(0, dw)
    ptop = random.randint(0, dh)
    pbot = random.randint(0, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    flip = random.randint(1, 10000) % 2
    # TODO remove to reactivate flip !!!!!!!!!!!!!!!!
    flip = 0
    img = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    sized = img.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, pleft, ptop, swidth, sheight


def fill_truth_detection(bboxes, final_width, final_height,
                         flip, pleft, ptop, swidth, sheight):
    new_bboxes = np.zeros(bboxes.shape)
    for i, bbox in enumerate(bboxes):
        # Obtain coordinates for scaled image
        ori_x1, ori_y1, ori_x2, ori_y2 = bbox
        ori_box_width = ori_x2 - ori_x1
        ori_box_height = ori_y2 - ori_y1
        crop_x1 = ori_x1 - pleft
        crop_y1 = ori_y1 - ptop

        # Compute values scaled by final_size/crop_size
        width_ratio = final_width / swidth
        scale_x1 = crop_x1 * width_ratio
        final_box_width = ori_box_width * width_ratio

        height_ratio = final_height / sheight
        scale_y1 = crop_y1 * height_ratio
        final_box_height = ori_box_height * height_ratio

        final_x1 = max(0, scale_x1)
        final_y1 = max(0, scale_y1)
        final_x2 = min(final_width, final_x1 + final_box_width)
        final_y2 = min(final_height, final_y1 + final_box_height)

        if flip:
            final_x1, final_x2 = final_width - final_x1, final_width - final_x2

        new_bboxes[i] = final_x1, final_y1, final_x2, final_y2
    return new_bboxes


def data_augmentation(img, bboxes, shape, jitter, hue, saturation, exposure):
    img, flip, pleft, ptop, swidth, sheight = image_augmentation(img, shape,
                                                                 jitter, hue,
                                                                 saturation,
                                                                 exposure)
    new_bboxes = fill_truth_detection(bboxes, img.width, img.height,
                                      flip, pleft, ptop, swidth, sheight)
    return img, new_bboxes


