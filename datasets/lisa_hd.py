import pickle
import os

import numpy as np
import scipy.sparse

# from functools import partial

from .imdb import ImageDataset
from .voc_eval import voc_eval
# from utils.yolo import preprocess_train


class LISADataset(ImageDataset):
    def __init__(self, split, datadir, batch_size, im_processor,
                 processes=3, shuffle=True, dst_size=None,
                 use_cache=False):
        """
        Args:
            split(str): either test or train
        """
        super(LISADataset, self).__init__('LISA' + split, datadir, batch_size,
                                          im_processor, processes, shuffle,
                                          dst_size)
        self.split = split
        self.use_cache = use_cache

        # Set usefull paths for given split
        self._data_path = os.path.join(datadir, 'LISA_HD')
        self._split_path = os.path.join(self._data_path, split)
        self._img_folder = os.path.join(self._split_path, 'pos')
        self._annot_folder = os.path.join(self._split_path, 'posGt')
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert os.path.exists(
            self._split_path), 'Split path does not exist:\
            {}'.format(self._split_path)

        self._classes = [('hand')]
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self.load_dataset()

    def load_dataset(self):
        # set self._image_index and self._annotations
        img_files = sorted(os.listdir(self._img_folder))
        img_paths = [os.path.join(self._img_folder, img_file)
                     for img_file in img_files]
        annot_paths = [os.path.join(self._annot_folder,
                                    img_file.split('.')[0] + '.txt')
                       for img_file in img_files]
        self._image_names = img_paths
        self._annot_paths = annot_paths
        self._image_indexes = range(len(self._image_names))
        self._annotations = self._load_annotations()

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

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
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        annot_path = self._annot_paths[index]
        # Bounding box as [x_min, y_min, width, height]
        bounding_boxes = np.loadtxt(annot_path, usecols=[1, 2, 3, 4],
                                    skiprows=1)
        if bounding_boxes.ndim == 1:
            bounding_boxes = np.expand_dims(bounding_boxes, axis=0)
        num_objs = bounding_boxes.shape[0]

        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        bounding_boxes[:, 2] = bounding_boxes[:, 0] + bounding_boxes[:, 2]
        bounding_boxes[:, 3] = bounding_boxes[:, 1] + bounding_boxes[:, 3]
        # Load object bounding boxes into a data frame.
        for ix, row in enumerate(bounding_boxes):
            cls_idx = self._class_to_ind['hand']
            gt_classes[ix] = cls_idx
            # Obtain max abs and ord from mins, weidth and height
            overlaps[ix, cls_idx] = 1.0
            seg_areas[ix] = (row[3] - row[1] + 1) * (row[2] - row[0] + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': bounding_boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + \
            self._image_set + '_{:s}.txt'
        filedir = os.path.join(
            self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_idx, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_indexes):
                    dets = all_boxes[cls_idx][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id
