from copy import deepcopy
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from constants import *

class AnchorFinder:

    PLOT_RESULTS = False

    def __init__(self, img_info):

        self.bbox = []
        '''
            bbox: [(w0, h0), (w1, h1), ...]
        '''

        for img_d in img_info["train"].values():
            for bbox_d in img_d["objs"]:

                self.bbox.append((bbox_d[3], bbox_d[4]))

        self.ROUNDS = 64
        '''
            arbitrary number of rounds for k means clustering
        '''

        self.k_means_results = []

    def get_anchors(self):
        '''
            call k-means and decide at which scale to use the anchors
        '''
        
        self.k_means()

        anchors_ = deepcopy(self.k_means_results)
        anchors_.sort(key=lambda dim: dim[0] + dim[1])
        anchors = [[] for _ in range(SCALE_CNT)]

        for d in range(SCALE_CNT):
            anchors[d] = anchors_[d * ANCHOR_PERSCALE_CNT: (d + 1) * ANCHOR_PERSCALE_CNT]

        return anchors

    def k_means(self):
        '''
            k-means clustering
        '''

        repeat_flag = False
        '''
            in case one anchor is initialized so far away that it does not converge to any answer
        '''

        def _distance(w1, h1, w2, h2):
        
            intersect_w = np.minimum(w1, w2)
            intersect_h = np.minimum(h1, h2)

            intersection = intersect_w * intersect_h
            union = w1 * h1 + w2 * h2 - intersection

            return 1 - intersection / union

        k = SCALE_CNT * ANCHOR_PERSCALE_CNT

        means = np.random.uniform(0, IMG_SIZE[0], size=(k, 2))

        for _ in range(self.ROUNDS):

            current_means = np.zeros((k, 2))
            current_cluster_count = np.zeros(k)

            for w, h in self.bbox:

                min_center_idx = None
                min_d = 2

                for center_idx in range(k):

                    dist = _distance(means[center_idx][0], means[center_idx][1], w, h)
                    if dist < min_d:

                        min_center_idx = center_idx
                        min_d = dist

                current_means[min_center_idx][0] += w
                current_means[min_center_idx][1] += h

                current_cluster_count[min_center_idx] += 1

            for center_idx in range(k):

                if np.isclose(current_cluster_count[center_idx], 0):
                    repeat_flag = True
                    break
                
                means[center_idx][0] = current_means[center_idx, 0] / current_cluster_count[center_idx]
                means[center_idx][1] = current_means[center_idx, 1] / current_cluster_count[center_idx]

            if repeat_flag is True:
                break

        self.k_means_results = means.tolist()

        if repeat_flag is True:
            self.k_means()

        # plotting - just for testing
        if not AnchorFinder.PLOT_RESULTS:
            return

        cluster_members = [[[], []] for _ in range(k)]

        for w, h in self.bbox:

            min_center_idx = None
            min_d = 2

            for center_idx in range(k):

                dist = _distance(means[center_idx][0], means[center_idx][1], w, h)
                if dist < min_d:

                    min_center_idx = center_idx
                    min_d = dist

            cluster_members[min_center_idx][0].append(w)
            cluster_members[min_center_idx][1].append(h)

        for center_idx, color in enumerate(["blue", "green", "red", "yellow", "purple", "brown", "cyan", "orange", "black"]):
            
            plt.scatter(cluster_members[center_idx][0], 
                        cluster_members[center_idx][1],
                        c=color,
                        marker='x')

            plt.scatter([means[center_idx][0]],
                        [means[center_idx][1]],
                        c=color,
                        marker="o",
                        linewidths=4)

        plt.show()
