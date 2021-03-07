import os
import sys
import random
import argparse
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from graspnet_dataset import GraspNetDataset
from around_view.utils.grasp_det import views2grasps
from around_view.utils.view_find import RandomViewSelector, FixedViewSelector, RNNViewSelector, RLViewSelector
from around_view.utils.grasp_mix import GraspMixer
from around_view.utils.evaluation import AroundViewGraspEval

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--max_view', type=int, default=3, help='view index: [0, 256)')
parser.add_argument('--method', required=True, help='the method of selecting views')
parser.add_argument('--num_workers', type=int, default=30, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Create Dataset for Scene_List
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test', camera=cfgs.camera, 
                               num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)
SCENE_LIST = TEST_DATASET.scene_list()
SCENE_LIST = sorted(set(SCENE_LIST), key=SCENE_LIST.index)

print(len(SCENE_LIST))


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def please_choose_your_hero(cfgs):
    if cfgs.method == 'random':
        agent = RandomViewSelector(cfgs)
    elif cfgs.method == 'fixed':
        agent = FixedViewSelector(cfgs)
    elif cfgs.method == 'rnn':
        agent = RNNViewSelector(cfgs)
    elif cfgs.method == 'rl':
        agent = RLViewSelector(cfgs)
    else:
        raise NameError('Invalid Method')
    return agent


def inference():
    agent = please_choose_your_hero(cfgs)
    mixer = GraspMixer()

    for scene_id in tqdm([int(x[-4:]) for x in SCENE_LIST]):
        views = agent.get_views()
        grasps_from_multi_views = views2grasps(scene_id, views, cfgs)
        grasp_group = mixer.mix_grasps(grasps_from_multi_views)

        save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[scene_id-100], cfgs.camera)
        save_path = os.path.join(save_dir, f'{cfgs.method}_views.npy')
        np.save(save_path, views)
        save_path = os.path.join(save_dir, f'{cfgs.method}_grasps.npy')
        grasp_group.save_npy(save_path)


def evaluate():
    ge = AroundViewGraspEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test', method=cfgs.method)
    res, ap = ge.eval_all(cfgs.dump_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, f'ap_{cfgs.method}.npy')
    np.save(save_dir, res)


if __name__ == '__main__':
    inference()
    evaluate()
