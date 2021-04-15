import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from around_view.utils.grasp import AroundViewGraspGroup

class GraspMixer():
    def __init__(self, *args):
        pass

    def mix_grasps(self, grasps):
        unit_len = len(grasps[0])
        grasp_group = AroundViewGraspGroup(ann_ids=np.zeros((0), dtype=int), camera_poses=grasps[0].camera_poses)
        for g in grasps:
            grasp_group.add(g)
        grasp_group.sort_by_score()
        # grasp_group = grasp_group.to_view(0)
        grasp_group = grasp_group[: unit_len]
        return grasp_group
