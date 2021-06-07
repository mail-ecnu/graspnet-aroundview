import numpy as np

import torch

from .pytorch_utils import to_var
from ..utils import GraspDetector, GraspMixer, AroundViewGraspEval
from ..utils.dataset import ALL_ANN_IDs


class LossComputer():
    def __init__(self, cfgs):
        self.detector = GraspDetector(cfgs.dataset_root, cfgs.dump_dir, cfgs.camera)
        self.mixer = GraspMixer()
        self.eval = AroundViewGraspEval(cfgs.dataset_root, cfgs.camera, 'train', cfgs.method)

    def get_loss(self, end_views):
        scene_ids = [int (s) for s in end_views[0]['scene']]
        batch_views = torch.cat((to_var([[0],[0]]), torch.cat([v['view'] for v in end_views], dim=1)) , dim=1)
        bs, seq_len = batch_views.shape

        loss = 0
        for b in range(bs):
            seq_views = list()
            seq_grasps = list()
            seq_accs = list()
            for s in range(seq_len):
                grasp_group = self.detector.views2grasps(scene_ids[b], batch_views[b][:s+1])
                grasp_group = self.mixer.mix_grasps(grasp_group)
                views = batch_views[b][:s+1]
                ann_ids = ALL_ANN_IDs[views.cpu()]
                acc = np.mean(self.eval.eval_scene(scene_ids[b], views=ann_ids, grasp_group=grasp_group)[0])
                

                # seq_views.append(views)
                # seq_grasps.append(grasp_group)
                seq_accs.append(acc)
            rewards = np.diff(seq_accs, n=1)
            loss += (-np.sum(rewards))

        return loss, end_views
