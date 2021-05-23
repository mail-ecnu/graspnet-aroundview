import os
import sys

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.backbone import Pointnet2Backbone


class PreRNN(nn.Module):
    def __init__(self, points_num, feat_dim, out_dim):
        super().__init__()

    def forward(self, seed_features):
        pass


class RNNController(nn.Module):
    def __init__(self, input_feature_dim=0):
        super().__init__()
        # see it later
        import ipdb; ipdb.set_trace()

        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.pre_rnn = nn.Sequential(
            nn.Linear(784, 100),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self.rnn = nn.LSTM(10, 20, 2)

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)

        import ipdb; ipdb.set_trace()
        
        return seed_features
