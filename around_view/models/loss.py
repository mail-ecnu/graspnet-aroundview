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

    def _compute_all_rewards(self, bs, seq_len, scene_ids, batch_views):
        rewards = list()
        for b in range(bs):
            seq_accs = list()
            for s in range(seq_len):
                grasp_group = self.detector.views2grasps(scene_ids[b], batch_views[b][:s+1])
                grasp_group = self.mixer.mix_grasps(grasp_group)
                views = batch_views[b][:s+1]
                ann_ids = ALL_ANN_IDs[views.cpu()]
                acc = np.mean(self.eval.eval_scene(scene_ids[b], views=ann_ids, grasp_group=grasp_group)[0])
                seq_accs.append(acc)
            rewards.append(np.diff(seq_accs, n=1))
        return np.array(rewards)

    def _compute_q_value(self, all_rewards):
        q_vals = list()
        for step in range(all_rewards.shape[1]):
            q_vals.append(np.sum(all_rewards[:, step:], axis=1))
        return to_var(np.array(q_vals).T)

    def get_loss(self, end_views):
        scene_ids = [int (s) for s in end_views[0]['scene']]
        batch_v_ids = torch.cat((to_var([[0],[0]]), torch.cat([v['v_id'] for v in end_views], dim=1)) , dim=1)
        batch_views = torch.cat((to_var([[0],[0]]), torch.cat([v['view'] for v in end_views], dim=1)) , dim=1)
        bs, seq_len = batch_views.shape
        all_rewards = self._compute_all_rewards(bs, seq_len, scene_ids, batch_views)
        all_q_value = self._compute_q_value(all_rewards)

        all_selected_logprobs = to_var(torch.empty(0))
        for step in range(seq_len-1):
            logprob = end_views[step]['prob']
            batch_actions = batch_v_ids[:, step:step+1]
            batch_q_value = all_q_value[:, step:step+1]

            selected_logprobs = batch_q_value * torch.gather(logprob, 1, batch_actions)
            all_selected_logprobs = torch.cat((all_selected_logprobs, selected_logprobs), 1)
        loss = -all_selected_logprobs.mean()
        return loss, end_views
