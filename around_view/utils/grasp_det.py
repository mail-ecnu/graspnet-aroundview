import os
import sys
import argparse

from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import get_scene_name

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from around_view.utils.grasp import AroundViewGraspGroup


class GraspDetector():
    def __init__(self, dataset_root, dump_dir, camera):
        self.dataset_root = dataset_root
        self.dump_dir = dump_dir
        self.camera = camera

    def get_grasp_group(self, scene_id, view_id):
        npy_file_path = os.path.join(self.dump_dir, get_scene_name(scene_id), self.camera, '%04d.npy' % (view_id,))
        camera_poses_path = os.path.join(self.dataset_root, 'scenes', get_scene_name(scene_id), self.camera, 'camera_poses.npy')
        grasp_group = AroundViewGraspGroup().from_npy(npy_file_path, camera_poses_path)
        return grasp_group

    def views2grasps(self, scene_id, views):
        grasps = list()
        for view_id in views:
            g = self.get_grasp_group(scene_id, view_id)
            grasps.append(g)
        return grasps
