import os
import sys
import numpy as np

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from models.backbone import Pointnet2Backbone
from around_view.models.pytorch_utils import to_var
from around_view.utils.view_find import ViewSelector
from around_view.utils.dataset import VIEW_LEN


class PreMLP(nn.Module):
    def __init__(self, points_num=256, feat_dim=1024, out_dim=512):
        super().__init__()
        self.w0 = Pointnet2Backbone()
        self.w1 = nn.Linear(points_num, 1)
        self.w2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, pointcloud):               # b, point, dim
        ''' x: seed_features of `b * [256, 1024]`
            out: features of `b * [out_dim]`
        '''
        point_feature, _, _ = self.w0(pointcloud)
        feature_point = point_feature.permute(0, 2, 1)          # b, dim_in, point
        feature = self.w1(feature_point).squeeze()        # b, dim_in
        return self.w2(feature)               # b, dim_out


class RNNController(nn.Module, ViewSelector):
    def __init__(self, cfgs, device):
        super().__init__()
        ViewSelector.__init__(self, cfgs)

        self.device = device
        # for views
        self.views_emb_size = 128
        # for points
        self.point_emb_size = 512
        self.hidden_size = 512

        # self.answer_seq_len = answer_seq_len
        # self.weight_size = weight_size

        self.enc = nn.Embedding(VIEW_LEN, self.views_emb_size)  # embed inputs(views)

        self.backbone = PreMLP(256, 1024, self.point_emb_size)
        # self.enc = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(self.views_emb_size, self.hidden_size)  # LSTMCell's input is always batch first

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
        bs = batch_data.shape[0]
        
        
        
        view_candidate = self.enc(to_var(torch.arange(VIEW_LEN).repeat(bs, 1)))
        selected_views = to_var(torch.empty((bs, 0)))

        v = to_var(self.first_view()).repeat(bs, 1)                 # [bs, 1]
        for i in range(self.max_view):
            selected_views = torch.cat((selected_views, v), 1)      # [bs, i+1]
            view_data = batch_data[torch.arange(bs), v.view(-1)]
            cloud_feats = self.backbone(view_data)

            import ipdb; ipdb.set_trace()
            print('rua')
            

        import ipdb; ipdb.set_trace()
        x_0 = 0
        h_0 = 0
        c_0 = 0

        return seed_features
