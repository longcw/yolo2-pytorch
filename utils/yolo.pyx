cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "math.h":
    double abs(double m)
    double log(double x)


def yolo_to_bbox(
        np.ndarray[DTYPE_t, ndim=2] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, int H, int W):
    return yolo_to_bbox_c(bbox_pred, anchors, H, W)

cdef yolo_to_bbox_c(
        np.ndarray[DTYPE_t, ndim=2] bbox_pred,
        np.ndarray[DTYPE_t, ndim=2] anchors, int H, int W):
    """
    Parameters
    ----------
    bbox_pred: (HxWxnum_anchors, 4) ndarray of float (sig(tx), sig(ty), exp(tw), exp(th))
    anchors: (num_anchors, 2) (pw, ph)
    Returns
    -------
    bbox_out: (HxWxnum_anchors, 4) ndarray of bbox (x1, y1, x2, y2) rescaled to (0, 1)
    """
    cdef unsigned int num_anchors = anchors.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] bbox_out = np.zeros((H*W*num_anchors, 4), dtype=DTYPE)

    cdef DTYPE_t cx, cy, bw, bh
    cdef unsigned int row, col, a, ind
    for row in range(H):
        for col in range(W):
            for a in range(num_anchors):
                ind = (row * W + col) * num_anchors + a
                cx = (bbox_pred[ind, 0] + col) / W
                cy = (bbox_pred[ind, 1] + row) / H
                bw = bbox_pred[ind, 2] * anchors[a][0] / W * 0.5
                bh = bbox_pred[ind, 3] * anchors[a][1] / H * 0.5

                bbox_out[ind, 0] = cx - bw
                bbox_out[ind, 1] = cy - bh
                bbox_out[ind, 2] = cx + bw
                bbox_out[ind, 3] = cy + bh

    return bbox_out