import os
import sys
import numpy as np

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.backbone import Pointnet2Backbone


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use a fake 'Kaiming normalization': sqrt(2 / fan_in)
    """
    fan_in = shape[0]  # here, is a fake 'fan_in'
    # randn is standard normal distribution generator. 
    w = torch.randn(shape, dtype=torch.float32) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


class PreMLP(nn.Module):
    def __init__(self, points_num=256, feat_dim=1024, out_dim=512):
        super().__init__()
        self.w1 = random_weight([points_num])
        self.w2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        ''' x: seed_features of `b * [256, 1024]`
            out: features of `b * [out_dim]`
        '''
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, self.pre_mat)
        return self.w2(x)


class RNNController(nn.Module):
    def __init__(self, device, input_feature_dim=0):
        super().__init__()
        self.device = device

        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.pre_MLP = PreMLP(256, 1024, 512)
        # import ipdb; ipdb.set_trace()

        # batch_first – If True, then the input and output tensors are 
        #               provided as (batch, seq, feature). Default: False
        self.h0 = 0
        self.c0 = 0
        self.rnn = nn.LSTM(512, 256, 2)

    def _rnn_step(end_points, hidden_state, cell):
        idx = -1
        seed_features, seed_xyz, end_points = self.backbone(pointcloud)
        x = self.pre_MLP(seed_features)
        # seed_features
        x, (h_t, c_t) = self.rnn(x, (hidden_state, cell))
        return idx

    def forward(self, point_clouds):
        init_view = torch.from_numpy(np.zeros(end_points.shape[0])).to(self.device)
        import ipdb; ipdb.set_trace()
        x_0 = 0
        h_0 = 0
        c_0 = 0

        import ipdb; ipdb.set_trace()
        return seed_features
