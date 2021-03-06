import numpy as np

import torch

from .pytorch_utils import to_var
from ..utils import GraspDetector, GraspMixer, AroundViewGraspEval
from ..utils.dataset import ALL_ANN_IDs


class LossComputer():
    def __init__(self, cfgs):
        self.author_tag = 'RL'

        self.detector = GraspDetector(cfgs.dataset_root, cfgs.dump_dir, cfgs.camera)
        self.mixer = GraspMixer()
        self.eval = AroundViewGraspEval(cfgs.dataset_root, cfgs.camera, 'train', cfgs.method)

    def _compute_all_rewards(self, bs, seq_len, scene_ids, batch_views, infos):
        batch_views = batch_views.cpu().numpy()
        
        rewards = list()
        eval_acc = 0
        for b in range(bs):
            seq_accs = list()
            for s in range(seq_len):
                grasp_group = self.detector.views2grasps(scene_ids[b], batch_views[b][:s+1])
                grasp_group = self.mixer.mix_grasps(grasp_group)
                views = batch_views[b][:s+1]
                acc = self.eval.continuous_eval_scene(scene_ids[b], views, grasp_group)
                seq_accs.append(acc)
            eval_acc += seq_accs[-1]
            rewards.append(np.diff(seq_accs, n=1))
        infos.update({f'{self.author_tag}/mAP': 1.0 * eval_acc / bs,})
        return np.array(rewards), infos

    def _compute_q_value(self, all_rewards):
        q_vals = list()
        for step in range(all_rewards.shape[1]):
            q_vals.append(np.sum(all_rewards[:, step:], axis=1))
        return to_var(np.array(q_vals).T)

    def get_loss(self, end_views):
        infos = dict()  # infos = dict(end_views=end_views)
        scene_ids = [int (s) for s in end_views[0]['scene']]
        batch_v_ids = torch.cat((to_var([[0],[0]]), torch.cat([v['v_id'] for v in end_views], dim=1)) , dim=1)
        batch_views = torch.cat((to_var([[0],[0]]), torch.cat([v['view'] for v in end_views], dim=1)) , dim=1)
        bs, seq_len = batch_views.shape
        all_rewards, infos = self._compute_all_rewards(bs, seq_len, scene_ids, batch_views, infos)
        all_q_value = self._compute_q_value(all_rewards)

        all_selected_logprobs = to_var(torch.empty(0))
        for _step in range(seq_len-1):
            logprob = end_views[_step]['prob']
            batch_actions = batch_v_ids[:, _step+1:_step+2]
            batch_q_value = all_q_value[:, _step+1:_step+2]

            try:
                selected_logprobs = batch_q_value * torch.gather(logprob, 1, batch_actions)
            except:
                import ipdb; ipdb.set_trace()
            all_selected_logprobs = torch.cat((all_selected_logprobs, selected_logprobs), 1)
        # print(all_selected_logprobs)
        assert len(all_selected_logprobs) > 0
        loss = -all_selected_logprobs.mean()

        try:
            loss_value = float(loss)
        except:
            # print(all_selected_logprobs)
            # print('loss: ', loss)
            import ipdb; ipdb.set_trace()

        infos.update({
            f'{self.author_tag}/loss': loss_value,
            f'{self.author_tag}/reward': float(np.sum(all_rewards, axis=1).mean()),
        })
        return loss, infos
