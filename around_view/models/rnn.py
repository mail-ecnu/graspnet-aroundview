import os
import sys
import numpy as np

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.backbone import Pointnet2Backbone
from around_view.models import pytorch_utils as ptu
from around_view.utils.view_find import ViewSelector
from around_view.utils.dataset import VIEW_LEN


class PreMLP(nn.Module):
    def __init__(self, points_num=256, feat_dim=1024, out_dim=512):
        super().__init__()
        self.w1 = nn.Linear(points_num, 1)
        self.w2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):               # b, point, dim
        ''' x: seed_features of `b * [256, 1024]`
            out: features of `b * [out_dim]`
        '''
        x = x.permute(0, 2, 1)          # b, dim_in, point
        x = self.w1(x).squeeze()        # b, dim_in
        return self.w2(x)               # b, dim_out


class RNNController(nn.Module, ViewSelector):
    def __init__(self, cfgs, device, input_feature_dim=0):
        super().__init__()
        ViewSelector.__init__(self, cfgs)

        self.device = device
        self.emb_size = 512
        self.hidden_size = 512

        # self.answer_seq_len = answer_seq_len
        # self.weight_size = weight_size
        
        # self.emb = nn.Embedding(input_size, emb_size)  # embed inputs

        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.preMLP = PreMLP(256, 1024, self.emb_size)        
        self.enc = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(self.emb_size, self.hidden_size)  # LSTMCell's input is always batch first

        self.hiedden = (torch.randn(1, 1, 3),
                        torch.randn(1, 1, 3))

    def _rnn_step(end_points, hidden_state, cell):
        idx = -1
        seed_features, seed_xyz, end_points = self.backbone(pointcloud)
        x = self.pre_MLP(seed_features)
        # seed_features
        x, (h_t, c_t) = self.rnn(x, (hidden_state, cell))
        return idx

    def forward(self, batch_data):
        self.first_view()
        batch_selected_mask = ptu.to_var(np.repeat(self.selected_mask[np.newaxis, :], batch_data.shape[0], axis=0))

        example_one_view_data = batch_data[:, 0, :, :]
        point_feats, _, _ = self.backbone(example_one_view_data)
        cloud_feats = self.preMLP(point_feats)

        import ipdb; ipdb.set_trace()
        x_0 = 0
        h_0 = 0
        c_0 = 0

        import ipdb; ipdb.set_trace()
        return seed_features
