import pickle
import os

import numpy as np
import scipy.sparse
import scipy.io as sio

# from functools import partial

from .imdb import ImageDataset
from .voc_eval import voc_eval
# from utils.yolo import preprocess_train


class EgoHandDataset(ImageDataset):
    def __init__(self, split, datadir, batch_size, im_processor,
                 processes=3, shuffle=True, dst_size=None,
                 differentiate_left_right=True):
        """
        Args:
            split(str): either test or train
        """
        super(EgoHandDataset, self).__init__('EgoHand' + split, datadir,
                                             batch_size, im_processor,
                                             processes, shuffle,
                                             dst_size)
        self.split = split

        # Set usefull paths for given split
        self._data_path = os.path.join(datadir, 'egohands')
        self._video_path = os.path.join(self._data_path, '_LABELLED_SAMPLES')
        self._annot_path = os.path.join(self._data_path, 'metadata.mat')
        assert os.path.exists(
            self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert os.path.exists(
            self._annot_path), 'Path does not exist:\
            {}'.format(self._annot_path)

        self.differentiate_left_right = differentiate_left_right
        if self.differentiate_left_right:
            self._classes = [('left_hand'), ('right_hand')]
        else:
            self._classes = [('hand')]
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.video_nb = 48
        self.frame_nb = 100  # labelled frames per video

        self.load_dataset()

    def get_image_path(self, video_idx, frame_idx,
                       img_file_template='frame_{frame:04d}.jpg'):
        """To be called after self.load_dataset()
        """
        video_folder = os.path.join(
            self._video_path, self.video_ids[video_idx])
        frames_annots = self.videos_annots[0, video_idx]
        frame_annots = frames_annots[0, frame_idx]
        frame_num = frame_annots['frame_num'][0, 0]
        frame_path = os.path.join(video_folder,
                                  img_file_template.format(frame=frame_num))
        assert os.path.exists(
            frame_path), 'file {} not found'.format(frame_path)
        return frame_path

    def load_dataset(self):
        metadata = sio.loadmat(self._annot_path)
        annots_frame = metadata['video']  # numpy.ndarray of shape (1, 48)
        nd_video_ids = annots_frame['video_id']
        self.videos_annots = annots_frame['labelled_frames']
        self.video_ids = [str(nd_video_ids[0, i][0])
                          for i in range(self.video_nb)]

        # set self._image_index and self._annotations
        self.idx_tuples = [(video_idx, video_frame)
                           for video_idx in range(self.video_nb)
                           for video_frame in range(self.frame_nb)]
        self._image_indexes = range(len(self.idx_tuples))

        img_paths = [self.get_image_path(video_idx, frame_idx)
                     for video_idx, frame_idx in self.idx_tuples]
        self._image_names = img_paths
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
        if os.path.exists(cache_file):
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
        video_idx, frame_idx = self.idx_tuples[index]
        bboxes, gt_classes = self.get_frame_annots(video_idx, frame_idx)
        num_objs = bboxes.shape[0]

        # if not self.differentiate_left_right:
        # Keep only one class for hands, zero labels for all
        #   gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, row in enumerate(bboxes):
            cls_idx = gt_classes[ix]
            # Obtain max abs and ord from mins, weidth and height
            overlaps[ix, cls_idx] = 1.0
            seg_areas[ix] = (row[3] - row[1] + 1) * (row[2] - row[0] + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': bboxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def get_frame_annots(self, video_idx, frame_idx):
        """Gets frame bounding boxes and left/right hand labels

            Returns:
                bbox(numpy.ndarray): each row is a [x_min, y_min, x_max, y_max]
                    bounding box
                hand_labels(numpy.ndarray): each value matches a bbox row with
                    1 for right and 0 for left hand
        """
        frames_annots = self.videos_annots[0, video_idx]
        frame_annots = frames_annots[0, frame_idx]

        def get_bb(frame_annots, segm_name):
            """Reads bbox from segmentation where frame_annots
            is the frame data_frame and segm_name is the key of
            the segmentation in the data_frame
            """
            my_left_seg = frame_annots[segm_name]
            if len(my_left_seg):
                x_min, y_min = my_left_seg.min(0)
                x_max, y_max = my_left_seg.max(0)
                bbox = np.array([int(x_min), int(y_min),
                                 int(x_max), int(y_max)])
            else:
                bbox = None
            return bbox

        # Left bboxes
        bbox_myleft = get_bb(frame_annots, 'myleft')
        bbox_yourleft = get_bb(frame_annots, 'yourleft')

        # Right bboxes
        bbox_myright = get_bb(frame_annots, 'myright')
        bbox_yourright = get_bb(frame_annots, 'yourright')

        # Remove empty bboxes and stack them together
        # ! ordering of bboxes matters in labeling !
        bboxes = [bbox_myleft, bbox_yourleft, bbox_myright, bbox_yourright]
        bboxes = [bbox for bbox in bboxes
                  if bbox is not None]
        if len(bboxes) == 0:
            # np_bboxes = None
            # labels = None
            np_bboxes = np.array([0, 0, 0, 0]).reshape(1, 4)
            labels = np.array([0])
        else:
            np_bboxes = np.stack(bboxes)

            # Retrieve labels as numpy array
            left_labels = [0 for bbox in [bbox_myleft, bbox_yourleft]
                           if bbox is not None]
            right_labels = [1 for bbox in [bbox_myright, bbox_yourright]
                            if bbox is not None]

            # Set labels to 0 for left, 1 for right or 0 for all if no side
            # differentiation
            if self.differentiate_left_right:
                labels = np.array(left_labels + right_labels)
            else:
                labels = np.zeros(len(bboxes)).astype(int)
            assert len(labels) == len(bboxes), 'label number {}\
                should match bbox count'.format(len(labels), len(bboxes))
        return np_bboxes, labels

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
