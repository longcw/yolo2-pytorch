import copy
import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


from datasets.utils.augmentation import data_augmentation
from datasets.utils import filesys


class LISADataset(Dataset):
    def __init__(self, split, datadir, transform=None,
                 transform_params=None,
                 use_cache=False):
        """
        Args:
            split(str): either test or train
            transform: transform to apply to image after joint
                transformations have been applied
            transform_params(dict): dictionnary containing the params
                for data augmentation with keys {'scale', 'hue', 'jitter',
                'exposure', 'saturation'}
        """
        super(LISADataset, self).__init__()
        self.split = split
        self.use_cache = use_cache
        self.transform = transform
        self.transform_params = transform_params

        # Set usefull paths for given split
        self._data_dir = datadir
        self.name = 'lisa' + split
        self._data_path = os.path.join(datadir, 'LISA_HD')
        self._split_path = os.path.join(self._data_path, split)
        self._img_folder = os.path.join(self._split_path, 'pos')
        self._annot_folder = os.path.join(self._split_path, 'posGt')
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert os.path.exists(
            self._split_path), 'Split path does not exist:\
            {}'.format(self._split_path)

        self.classes = [('hand')]
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self.load_dataset()

    def load_dataset(self):
        img_files = sorted(os.listdir(self._img_folder))
        img_paths = [os.path.join(self._img_folder, img_file)
                     for img_file in img_files]
        annot_paths = [os.path.join(self._annot_folder,
                                    img_file.split('.')[0] + '.txt')
                       for img_file in img_files]
        self.image_names = img_paths
        self.annot_paths = annot_paths
        self.sample_nb = len(self.image_names)
        self.image_indexes = range(self.sample_nb)
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up
        future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file) and not self.use_cache:
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name,
                                                      cache_file))
            return roidb

        gt_roidb = [self._annotation_from_index(index)
                    for index in self.image_indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def __getitem__(self, idx):
        annotations = copy.deepcopy(self.annotations[idx])
        image_path = self.image_names[idx]
        img = Image.open(image_path)
        original_img = np.array(img)

        # Target/img joint transforms
        if self.transform_params is not None:
            final_shape = self.transform_params['shape']
            jitter = self.transform_params['jitter']
            hue = self.transform_params['hue']
            saturation = self.transform_params['saturation']
            exposure = self.transform_params['exposure']

            img, gt_boxes = data_augmentation(img, annotations['boxes'].copy(),
                                              final_shape, jitter, hue,
                                              saturation, exposure)
            annotations['boxes'] = gt_boxes

        if self.transform is not None:
            img = self.transform(img)
        return img, original_img, annotations

    def __len__(self):
        return self.sample_nb

    def _annotation_from_index(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        annot_path = self.annot_paths[index]
        # Bounding box as [x_min, y_min, width, height]
        bounding_boxes = np.loadtxt(annot_path, usecols=[1, 2, 3, 4],
                                    skiprows=1)
        if bounding_boxes.ndim == 1:
            bounding_boxes = np.expand_dims(bounding_boxes, axis=0)
        num_objs = bounding_boxes.shape[0]

        gt_classes = np.zeros((num_objs), dtype=np.int32)

        bounding_boxes[:, 2] = bounding_boxes[:, 0] + bounding_boxes[:, 2]
        bounding_boxes[:, 3] = bounding_boxes[:, 1] + bounding_boxes[:, 3]
        # Load object bounding boxes into a data frame.
        for ix, row in enumerate(bounding_boxes):
            cls_idx = self._class_to_ind['hand']
            gt_classes[ix] = cls_idx

        return {'boxes': bounding_boxes,
                'gt_classes': gt_classes,
                'flipped': False}

    @property
    def cache_path(self):
        cache_path = os.path.join(self._data_dir, 'cache')
        filesys.mkdir(cache_path)
        return cache_path

