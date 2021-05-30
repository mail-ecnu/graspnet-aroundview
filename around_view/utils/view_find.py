import os
import sys
import random
import numpy as np
from abc import ABCMeta, abstractmethod

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from around_view.utils.dataset import VIEW_LEN, ALL_ANN_IDs


class ViewSelector():
    __metaclass__ = ABCMeta

    def __init__(self, cfgs):
        self.views_len = VIEW_LEN
        self.all_ann_ids = ALL_ANN_IDs  # [0, 16, 32, 48...)
        self.selected_mask = np.zeros(self.views_len)
        self.selected_views = list()

        self.max_view = cfgs.max_view

    def finished(self):
        return len(self.selected_views) >= self.max_view

    def first_view(self):
        idx = 0  # random.randint(0, self.views_len-1)
        self.selected_mask[idx] = 1
        self.selected_views.append(idx)
        return idx

    @abstractmethod
    def next_view(self):
        raise NotImplementedError

    def get_views(self):
        self.first_view()
        while not self.finished():
            self.next_view()
        return self.all_ann_ids[self.selected_views]


class RandomViewSelector(ViewSelector):
    def next_view(self):
        idx = -1
        while idx == -1 or idx in self.selected_views:
            idx = random.randint(0, self.views_len-1)
        # idx = 1
        self.selected_mask[idx] = 1
        self.selected_views.append(idx)
        return idx


class FixedViewSelector(ViewSelector):
    def get_views(self):
        self.selected_views = [i for i in range(self.views_len)]
        return self.all_ann_ids[self.selected_views]


class RNNViewSelector(ViewSelector):
    def next_view(self):
        # TODO
        raise NotImplementedError


class RLViewSelector(ViewSelector):
    def next_view(self):
        # TODO
        raise NotImplementedError


class SeqViewSelector(ViewSelector):

    def __init__(self, cfgs):
        self.views_len = 256
        self.all_ann_ids = np.array([x for x in range(self.views_len)])
        self.max_view = 256

    def get_views(self):
        return self.all_ann_ids[:self.max_view]
