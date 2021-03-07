import os
import argparse

from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import get_scene_name


def get_grasp(scene_id, view_id, dump_dir, camera):
    grasp_group = GraspGroup().from_npy(os.path.join(dump_dir,get_scene_name(scene_id), camera, '%04d.npy' % (view_id,)))
    return grasp_group


def views2grasps(scene_id, views, cfgs):
    grasps = list()
    for view_id in views:
        g = get_grasp(scene_id, view_id, cfgs.dump_dir, cfgs.camera)
        grasps.append(g)
    return grasps
