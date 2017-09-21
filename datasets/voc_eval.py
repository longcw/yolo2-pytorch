# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import pdb


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects


def voc_ap(recall, prec, use_07_metric=False):
    """ ap = voc_ap(recall, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(prec[recall >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             use_cache=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # Load ground truth annotations
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile) or not use_cache:
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            # if i % 100 == 0:
            #    print('Reading annotation for {:d}/{:d}'.format(
            #        i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + len(bbox)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [line.strip().split(' ') for line in lines]
        image_ids = [line[0] for line in splitlines]
        confidence = np.array([float(line[1]) for line in splitlines])
        BB = np.array([[float(z) for z in line[2:]] for line in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down detections and mark TPs and FPs
        image_nb = len(image_ids)
        true_positives = np.zeros(image_nb)
        false_positives = np.zeros(image_nb)
        for image_idx in range(image_nb):
            R = class_recs[image_ids[image_idx]]
            bb = BB[image_idx, :].astype(float)
            max_overlap = -np.inf
            BBGT = R['bbox'].astype(float)

            # Compute intersection over union area
            if BBGT.size > 0:
                # Compute intersection bounding box
                inter_xmin = np.maximum(BBGT[:, 0], bb[0])
                inter_ymin = np.maximum(BBGT[:, 1], bb[1])
                inter_xmax = np.minimum(BBGT[:, 2], bb[2])
                inter_ymax = np.minimum(BBGT[:, 3], bb[3])

                # Compute intersection area
                inter_width = np.maximum(inter_xmax - inter_xmin + 1., 0.)
                inter_height = np.maximum(inter_ymax - inter_ymin + 1., 0.)
                inter_area = inter_width * inter_height

                # Compute union area
                union_area = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                              (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                              (BBGT[:, 3] - BBGT[:, 1] + 1.) - inter_area)

                overlap_ratios = inter_area / union_area
                max_overlap = np.max(overlap_ratios)
                jmax = np.argmax(overlap_ratios)

            if max_overlap > ovthresh:
                # if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    # Here image_idx matches a given (image, bbox) detection
                    true_positives[image_idx] = 1.
                    R['det'][jmax] = 1
                else:
                    false_positives[image_idx] = 1.
            else:
                false_positives[image_idx] = 1.

        # compute precision recall
        cumul_false_positives = np.cumsum(false_positives)
        cumul_true_positives = np.cumsum(true_positives)
        recall = cumul_true_positives / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = cumul_true_positives / np.maximum(cumul_true_positives +
                                                 cumul_false_positives,
                                                 np.finfo(np.float64).eps)
        ap = voc_ap(recall, prec, use_07_metric)
    else:
        recall = -1
        prec = -1
        ap = -1

    return recall, prec, ap
