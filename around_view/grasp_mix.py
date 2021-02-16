from graspnetAPI.grasp import GraspGroup


class GraspMixer():
    def __init__(self, *args):
        pass

    def mix_grasps(self, grasps):
        unit_len = len(grasps[0])

        grasp_group = GraspGroup()
        for g in grasps:
            grasp_group.add(g)
        grasp_group.sort_by_score()
        grasp_group = grasp_group[: unit_len]
        return grasp_group
