from collections import Counter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


def plot_bboxes(img, bboxes, scores=None):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i, row in enumerate(bboxes):
        xy = (row[0], row[1])
        w = row[2] - row[0]
        h = row[3] - row[1]
        detection_rect = Rectangle(xy, w, h,
                                   edgecolor='r',
                                   linewidth=1, facecolor='None')
        ax.add_patch(detection_rect)
        ax.text(xy[0], xy[1] - 2, 'hand : {:.3f}'.format(scores[i][0]),
                color='red')
    plt.show()
    return fig


def plot_hist(labels, proportion=True):
    """Plots histogram of labels where labels is a list of labels
    where each class is repeated as many times as it is present in
    the dataset
    """
    names, counts = compute_freqs(labels, proportion=proportion)
    indexes = np.arange(len(names))
    plt.bar(indexes, counts, tick_label=names)
    plt.xticks(rotation=90)
    plt.show()


def compute_freqs(labels, proportion=True):
    label_counter = Counter(labels).items()
    label_counter = sorted(label_counter, key=itemgetter(1))
    names, counts = zip(*label_counter)
    if proportion:
        sum_count = sum(counts)
        counts = [count/sum_count for count in counts]
    return names, counts


def draw2d_annotated_img(img, annot, links, keep_joints=None):
    """
    Draws 2d image img with joint annotations

    :param annot: First axes represent joint indexes
    second the u, v  (and useless d) joint coordinates
    :type annot: numpy ndarray
    :param keep_joints: only draws links between joints in keep_joints
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(img)
    ax.scatter(annot[:, 0], annot[:, 1], s=4, c="r")
    if(links):
        draw2djoints(ax, annot, links, keep_joints)


def draw3d_annotated_img(annot, links, keep_joints=None, angle=320):
    """
    Draws 3d image img with joint annotations

    :param annot: First axes represent joint indexes
    second the x, y, z joint coordinates
    :type annot: numpy ndarray
    :param keep_joints: only draws links between joints in keep_joints
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, angle)
    draw3djoints(ax, annot, links, keep_joints)


def draw2djoints(ax, annots, links, keep_joints=None):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot plot/subplot
    :param links: tuples of annot rows to link [(idx1, idx2), ...]
    :param keep_joints: only draws links between joints in keep_joints
    """
    for link in links:
        if keep_joints is None or (link[0] in keep_joints
                                   and link[1] in keep_joints):
            draw2dseg(ax, annots, link[0], link[1])


def draw2dseg(ax, annot, idx1, idx2, color="r", marker="o"):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot plot/subplot
    :param idx1: row of the start point in annot
    :param idx2: row of the end point in annot
    """
    ax.plot([annot[idx1, 0], annot[idx2, 0]],
            [annot[idx1, 1], annot[idx2, 1]],
            c=color, marker=marker)


def draw3djoints(ax, annots, links, keep_joints=None):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot 3d plot/subplot
    :param links: tuples of annot rows to link [(idx1, idx2), ...]
    :param keep_joints: only draws links between joints in keep_joints
    """
    for link in links:
        if keep_joints is None or (link[0] in keep_joints
                                   and link[1] in keep_joints):
            draw3dseg(ax, annots, link[0], link[1])


def draw3dseg(ax, annot, idx1, idx2, color="r"):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot 3d plot/subplot
    :param idx1: row of the start point in annot
    :param idx2: row of the end point in annot
    """
    ax.plot([annot[idx1, 0], annot[idx2, 0]],
            [annot[idx1, 1], annot[idx2, 1]],
            [annot[idx1, 2], annot[idx2, 2]])

