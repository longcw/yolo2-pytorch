import copy
import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


from datasets.utils.augmentation import data_augmentation
from datasets.utils import filesys


class SyntheticDataset(Dataset):
    def __init__(self, datadir, transform,
                 transform_params=None,
                 use_cache=False):
        """
        Args:
            split(str): either test or train
        """
        super(SyntheticDataset, self).__init__()
        self.use_cache = use_cache
        self.transform = transform
        self.transform_params = transform_params

        self.classes = [('hand')]
        self.num_classes = len(self.classes)

        # Set usefull paths for given split
        self._data_dir = datadir
        self.name = 'synthetic'
        self._data_path = os.path.join(datadir, 'synthetic')
        self._img_path = os.path.join(self._data_path, 'rgb')
        self._annot_path = os.path.join(self._data_path, '2Dcoord')
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert os.path.exists(
            self._annot_path), 'Path does not exist:\
            {}'.format(self._annot_path)

        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.load_dataset()
        self.num_samples = len(self.image_names)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        annotations = copy.deepcopy(self.annotations[idx])
        image_path = self.image_names[idx]
        img = Image.open(image_path)
        original_img = np.array(img)

        # Data augmentation for image
        if self.transform_params is not None:
            final_shape = self.transform_params['shape']
            jitter = self.transform_params['jitter']
            hue = self.transform_params['hue']
            saturation = self.transform_params['saturation']
            exposure = self.transform_params['exposure']
            img, annotations['boxes'] = data_augmentation(img,
                                                          annotations['boxes'].copy(
                                                          ),
                                                          final_shape, jitter, hue,
                                                          saturation, exposure)
        if self.transform is not None:
            img = self.transform(img)
        return img, original_img, annotations

    def load_dataset(self):
        hand_imgs = sorted(os.listdir(self._img_path))
        self.image_names = [os.path.join(self._img_path, img_name)
                            for img_name in hand_imgs]
        self.annotation_paths = [os.path.join(self._annot_path,
                                              img_name.split('.')[0] + '.txt')
                                 for img_name in hand_imgs]

        assert len(self.image_names) == len(self.annotation_paths)
        self.image_indexes = range(len(self.image_names))
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

    def _annotation_from_index(self, index):
        """
        Load bounding boxes info from .mat metadata file
        """
        annot_path = self.annotation_paths[index]
        bboxes = get_frame_annots(annot_path)

        # 0 class for all
        gt_classes = np.zeros(len(bboxes)).astype(int)

        # Return annots
        return {'boxes': bboxes,
                'gt_classes': gt_classes,
                'flipped': False}

    @property
    def cache_path(self):
        cache_path = os.path.join(self._data_dir, 'cache')
        filesys.mkdir(cache_path)
        return cache_path


def get_frame_annots(annot_path):
    """Gets frame bounding boxes
        Returns:
            bbox(numpy.ndarray): each row is a [x_min, y_min, x_max, y_max]
                bounding box
    """
    joints = np.loadtxt(annot_path)
    # Flip for synthetic dataset only!
    joints[:, 1] = 540 - joints[:, 1]
    bbox = np.array(bbox_from_joints(joints)).reshape(1, 4)
    return bbox


def bbox_from_joints(annots):
    """Retrieves tight bounding box around joints

    Args:
        annots (numpy.ndarray): array of dim nx2 where n is the nb of joints
            in format [[x_1, x_2, ...], [y_1, y_2, ...]]

    Returns:
        bbox (list): in format [x_min, y_min, x_max, y_max]
    """
    min_x, min_y = np.min(annots, axis=0)
    max_x, max_y = np.max(annots, axis=0)
    return [min_x, min_y, max_x, max_y]
