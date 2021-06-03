import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    '''https://github.com/jojonki/Pointer-Networks
    '''
    def __init__(self, cfgs, device):
        super().__init__()
        ViewSelector.__init__(self, cfgs)

        self.device = device
        self.views_emb_size = 128
        self.point_emb_size = 512
        self.hidden_size = 512
        self.weight_size = 256

        self.enc = nn.Embedding(VIEW_LEN, self.views_emb_size)  # embed views

        self.backbone = PreMLP(256, 1024, self.point_emb_size)  # embed points
        self.dec = nn.LSTMCell(self.point_emb_size + self.views_emb_size, self.hidden_size)

        self.W1 = nn.Linear(self.views_emb_size, self.weight_size, bias=False)  # blending encoder(left_view)
        self.W2 = nn.Linear(self.hidden_size, self.weight_size, bias=False)     # blending decoder(this->view + points)
        self.vt = nn.Linear(self.weight_size, 1, bias=False)                    # scaling sum of enc and dec by v.T

    def forward(self, batch_data):
        bs = batch_data.shape[0]
        bs_id = to_var(torch.arange(bs))
                                  
        # all_view_pool = to_var(torch.arange(VIEW_LEN)).repeat(bs, 1)
        all_view_pool = np.arange(VIEW_LEN)[np.newaxis, :].repeat(bs, axis=0)
        all_view_embb = self.enc(to_var(all_view_pool))

        selected_view = to_var(torch.empty((bs, 0))).type(torch.int64)
        v = to_var(self.first_view()).repeat(bs, 1)                 # [bs, 1]
        hidden = to_var(torch.zeros([bs, self.hidden_size]))        # (bs, h)
        cell_state = to_var(torch.zeros([bs, self.hidden_size]))    # (bs, h)

        probs = list()
        for i in range(self.max_view):
            selected_view = torch.cat((selected_view, v), 1)      # [bs, i+1]
            left_idxs, left_views = self._get_left_views(all_view_pool, all_view_embb, selected_view)
            blend1 = self.W1(left_views)

            view_data = batch_data[bs_id, v.view(-1)]
            view_feat = self.backbone(view_data)
            view_embb = all_view_embb[bs_id, v.view(-1)]
            dec_input = torch.cat((view_feat, view_embb), 1)
            hidden, cell_state = self.dec(dec_input, (hidden, cell_state))
            blend2 = self.W2(hidden)

            blend_sum = F.tanh(blend1 + blend2.unsqueeze(1).repeat(1, left_idxs.shape[1], 1))   # (L, bs, W)
            out = self.vt(blend_sum).squeeze()                                                  # (L, bs)
            out = F.log_softmax(out)                                                            # (bs, L)
            v = to_var(left_idxs)[bs_id, out.argmax(dim=1)].view(bs, -1)
            probs.append(dict(view=v, idxs=left_idxs, prob=out))

        return probs
    
    def _get_left_views(self, all_view_pool, all_view_embb, selected_view):
        bs, selected_num = selected_view.shape
        selected_view = selected_view.cpu()

        left_idxs = []
        left_views = to_var(torch.empty((0, VIEW_LEN - selected_num, self.views_emb_size)))
        for b in range(all_view_pool.shape[0]):
            idxs = np.setdiff1d(all_view_pool[b], selected_view[b])
            left_idxs.append(idxs)
            left_views = torch.cat((left_views, all_view_embb[b][idxs].unsqueeze(0)), 0)
        return np.array(left_idxs), left_views
