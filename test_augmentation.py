import cfgs.config as cfg
from datasets.lisa_hd import LISADataset
from datasets.utils.augmentation import data_augmentation
from datasets.utils.visualize import plot_bboxes

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

plot_bboxes(img, bboxes)

# Transform img and bboxes
new_img, new_bboxes = data_augmentation(img, bboxes, shape, jitter, hue,
                                        saturation, exposure)
plot_bboxes(new_img, new_bboxes)

