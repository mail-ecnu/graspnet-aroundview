import os
import numpy as np
from collections import Counter
from graspnetAPI.grasp import Grasp, GraspGroup


class AroundViewGrasp(Grasp):
    def __init__(self, group, *args):
        '''
        args[:-1]: args for Grasp
        args[-1]: ann_id
        '''
        super().__init__(*args[:-1])
        self._group = group
        self.ann_id = args[-1]

    def __repr__(self):
        return super().__repr__() + f'\nann_id: {self.ann_id}'

    @property
    def ann_id(self):
        return self._ann_id
    
    @ann_id.setter
    def ann_id(self, idx):
        assert idx in range(256)
        self._ann_id = idx

    def to_view(self, target_ann_id):
        T = self._group.convert_ann_id_matrix(self.ann_id, target_ann_id)
        self.transform(T)
        self.ann_id = target_ann_id


class AroundViewGraspGroup(GraspGroup):

    def from_npy(self, npy_file_path, camera_poses_path):
        super().from_npy(npy_file_path)
        ann_id = int(os.path.splitext(os.path.basename(npy_file_path))[0])
        self._ann_ids = np.full(self.__len__(), ann_id)
        self._camera_poses = np.load(camera_poses_path)
        return self

    def __repr__(self):
        repr = '----------\nGrasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for grasp_array, ann_id in zip(self.grasp_group_array, self.ann_ids):
                repr += AroundViewGrasp(self, grasp_array, ann_id).__repr__() + '\n'
        else:
            for i in range(3):
                repr += AroundViewGrasp(self, self.grasp_group_array[i], self.ann_ids[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += AroundViewGrasp(self, self.grasp_group_array[-(3-i)], self.ann_ids[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    @property
    def ann_ids(self):
        return self._ann_ids

    @ann_ids.setter
    def ann_ids(self, ids):
        assert len(ids) == self.__len__()
        self._ann_ids = ids
    
    @property
    def camera_poses(self):
        return self._camera_poses

    def convert_ann_id_matrix(self, original_ann_id, target_ann_id):
        original_camera_pose = self._camera_poses[original_ann_id]
        target_camera_pose = self._camera_poses[target_ann_id]
        return np.matmul(np.linalg.inv(target_camera_pose), original_camera_pose)

    def to_view(self, target_ann_id):
        if Counter(self.ann_ids)[self.ann_ids[0]] == self.__len__():
            # all grasps are in the same ann_id.
            T = self.convert_ann_id_matrix(self.ann_ids[0], target_ann_id)
            self.transform(T)
            self.ann_ids[:] = target_ann_id
            return self
        else:
            # for-loop, but currently we dont need this shit.
            raise NotImplementedError('Lazy cww')
