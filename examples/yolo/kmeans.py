# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

"""Kmeans algorithm to select Anchor shape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np


def parse_args():
    """
    Parse command-line flags passed to the training script.

    Returns:
      Namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(prog='kmeans', description='Kmeans to select anchors.')

    parser.add_argument(
        '-l',
        '--label_folders',
        type=str,
        required=True,
        nargs='+',
        help='Paths to label files')
    parser.add_argument(
        '-n',
        '--num_clusters',
        type=int,
        default=9,
        help='Number of clusters needed.'
    )
    parser.add_argument(
        '--ratio_x',
        type=float,
        default=1.0,
        help='x = ratio_x * kitti_x.'
    )
    parser.add_argument(
        '--ratio_y',
        type=float,
        default=1.0,
        help='y = ratio_y * kitti_y.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='y = ratio_y * kitti_y.'
    )
    parser.add_argument(
        '--min_x',
        type=float,
        default=10,
        help='ignore boxes with width (as in label) not larger than this value.'
    )
    parser.add_argument(
        '--min_y',
        type=float,
        default=10,
        help='ignore boxes with height (as in label) not larger than this value.'
    )
    return parser.parse_args()


def read_boxes(folders, ratio_x, ratio_y, min_x, min_y):
    '''
    Read all boxes as two numpy arrays.

    Args:
        folders (list of strings): paths to kitti label txts.
    Returns:
        w (1-d array): widths of all boxes
        h (1-d array): heights of all boxes
    '''

    w = []
    h = []
    for folder in folders:
        for txt in os.listdir(folder):
            _, ext = os.path.splitext(txt)
            if ext != '.txt':
                continue
            lines = open(os.path.join(folder, txt), 'r').read().split('\n')
            for l in lines:
                l_sp = l.strip().split()
                if len(l_sp) < 15:
                    continue
                left = float(l_sp[4])
                top = float(l_sp[5])
                right = float(l_sp[6])
                bottom = float(l_sp[7])
                l_w = right - left
                l_h = bottom - top

                if l_w > min_x and l_h > min_y:
                    w.append(l_w * ratio_x)
                    h.append(l_h * ratio_y)

    return np.array(w), np.array(h)


def iou(w0, h0, w1, h1):
    '''
    Pairwise IOU.

    Args:
        w0, h0: Boxes group 0
        w1, h1: Boxes group 1
    Returns:
        iou (len(w0) rows and len(w1) cols): pairwise iou scores
    '''
    len0 = len(w0)
    len1 = len(w1)
    w0_m = w0.repeat(len1).reshape(len0, len1)
    h0_m = h0.repeat(len1).reshape(len0, len1)
    w1_m = np.tile(w1, len0).reshape(len0, len1)
    h1_m = np.tile(h1, len0).reshape(len0, len1)
    area0_m = w0_m * h0_m
    area1_m = w1_m * h1_m
    area_int_m = np.minimum(w0_m, w1_m) * np.minimum(h0_m, h1_m)

    return area_int_m / (area0_m + area1_m - area_int_m)


def kmeans(w, h, num_clusters, max_steps=1000):
    '''
    Calculate cluster centers.

    Args:
        w (1-d numpy array): widths
        h (1-d numpy array): heights
        num_clusters (int): num clusters needed
    Returns:
        cluster_centers (list of tuples): [(c_w, c_h)] sorted by area
    '''

    assert len(w) == len(h), "w and h should have same shape"
    assert num_clusters < len(w), "Must have more boxes than clusters"
    n_box = len(w)
    rand_id = np.random.choice(n_box, num_clusters, replace=False)
    clusters_w = w[rand_id]
    clusters_h = h[rand_id]

    # EM-algorithm
    cluster_assign = np.zeros((n_box,), int)

    for i in range(max_steps):
        # shape (n_box, num_cluster)
        if i % 10 == 0:
            print("Start optimization iteration:", i + 1)
        box_cluster_iou = iou(w, h, clusters_w, clusters_h)
        re_assign = np.argmax(box_cluster_iou, axis=1)
        if all(re_assign == cluster_assign):
            # converge
            break
        cluster_assign = re_assign
        for j in range(num_clusters):
            clusters_w[j] = np.median(w[cluster_assign == j])
            clusters_h[j] = np.median(h[cluster_assign == j])

    return sorted(zip(clusters_w, clusters_h), key=lambda x: x[0] * x[1])


def main():
    '''Main function.'''

    args = parse_args()
    w, h = read_boxes(args.label_folders, args.ratio_x, args.ratio_y, args.min_x, args.min_y)
    results = kmeans(w, h, args.num_clusters, args.max_steps)
    print('Please use following anchor sizes in YOLO config:')
    for x in results:
        print("(%0.2f, %0.2f)" % x)


if __name__ == "__main__":
    main()
