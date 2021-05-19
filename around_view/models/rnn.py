import os
import sys

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

class RNNController(nn.Module):
    def __init__(self):
        # see it later
        import ipdb; ipdb.set_trace()

    def forward(self, rua):
        return None
