import os
import argparse

from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import get_scene_name


def get_grasp(scene_id, view_id, dump_dir, camera):
    grasp_group = GraspGroup().from_npy(os.path.join(dump_dir,get_scene_name(scene_id), camera, '%04d.npy' % (view_id,)))
    return grasp_group


'''
CUDA_VISIBLE_DEVICES=0 python grasp_det.py \
    --dump_dir ../logs/dump_rs \
    --camera realsense \
    --scene_id 100 \
    --view_id 0
'''
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
    parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
    parser.add_argument('--scene_id', required=True, type=int, help='scene index: [0: 190)')
    parser.add_argument('--view_id', required=True, type=int, help='view index: [0, 256)')
    # parser.add_argument('--num_workers', type=int, default=30, help='Number of workers used in evaluation [default: 30]')
    cfgs = parser.parse_args()

    grasp_group = get_grasp(scene_id=cfgs.scene_id, view_id=cfgs.view_id, dump_dir=cfgs.dump_dir, camera=cfgs.camera)
