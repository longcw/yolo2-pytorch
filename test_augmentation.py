from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cfgs.config as cfg
from datasets.lisa_hd import LISADataset
from datasets.utils.augmentation import data_augmentation

imdb = LISADataset('train', cfg.DATA_DIR, transform=None,
                   use_cache=False)


img, ori_img, targets = imdb[0]
bboxes = targets['boxes']

shape = (cfg.inp_size)
jitter = 0.1
hue = 0.1
saturation = 1.5
exposure = 1.5
print('old size ', img.height, img.width)

fig, ax = plt.subplots(1)
ax.imshow(img)
for row in bboxes:
    xy = (row[0], row[1])
    w = row[2] - row[0]
    h = row[3] - row[1]
    detection_rect = Rectangle(xy, w, h,
                               edgecolor='r',
                               linewidth=1, facecolor='None')
    ax.add_patch(detection_rect)
plt.show()

# Transform img and bboxes
new_img, new_bboxes = data_augmentation(img, bboxes, shape, jitter, hue,
                                        saturation, exposure)
fig, ax = plt.subplots(1)
ax.imshow(new_img)
print('new size ', new_img.height, new_img.width)
for row in new_bboxes:
    xy = (row[0], row[1])
    w = row[2] - row[0]
    h = row[3] - row[1]
    detection_rect = Rectangle(xy, w, h,
                               edgecolor='r',
                               linewidth=1, facecolor='None')
    ax.add_patch(detection_rect)
plt.show()
