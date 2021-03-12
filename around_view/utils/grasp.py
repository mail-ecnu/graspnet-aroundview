import copy
import os
import numpy as np
from collections import Counter
from graspnetAPI.grasp import Grasp, GraspGroup


class AroundViewGrasp(Grasp):
    def __init__(self, group, ann_id, *args):
        '''
        args[:-1]: args for Grasp
        args[-1]: ann_id
        '''
        super().__init__(*args)
        self._group = group
        self._ann_id = ann_id

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
        return self


class AroundViewGraspGroup(GraspGroup):

    def __init__(self, ann_ids=None, camera_poses=None, *args):
        super().__init__(*args)
        self._ann_ids = copy.deepcopy(ann_ids)
        self.camera_poses = copy.deepcopy(camera_poses)

    def from_npy(self, npy_file_path, camera_poses_path):
        super().from_npy(npy_file_path)
        ann_id = int(os.path.splitext(os.path.basename(npy_file_path))[0])
        self._ann_ids = np.full(self.__len__(), ann_id)
        self.camera_poses = np.load(camera_poses_path)
        return self

    def __repr__(self):
        repr = '----------\nGrasp Group, Number={}:\n'.format(self.__len__())
        if self.__len__() <= 6:
            for grasp_array, ann_id in zip(self.grasp_group_array, self.ann_ids):
                repr += AroundViewGrasp(self, ann_id, grasp_array).__repr__() + '\n'
        else:
            for i in range(3):
                repr += AroundViewGrasp(self, self.ann_ids[i], self.grasp_group_array[i]).__repr__() + '\n'
            repr += '......\n'
            for i in range(3):
                repr += AroundViewGrasp(self, self.ann_ids[-(3-i)], self.grasp_group_array[-(3-i)]).__repr__() + '\n'
        return repr + '----------'

    def __getitem__(self, index):
        '''
        **Input:**
        - index: int, slice, list or np.ndarray.

        **Output:**
        - if index is int, return Grasp instance.
        - if index is slice, np.ndarray or list, return GraspGroup instance.
        '''
        if type(index) == int:
            return AroundViewGrasp(self, self.ann_ids[index], self.grasp_group_array[index])
        elif type(index) == slice:
            graspgroup = AroundViewGraspGroup(self.ann_ids[index], self.camera_poses)
            graspgroup.grasp_group_array = copy.deepcopy(self.grasp_group_array[index])
            return graspgroup
        # elif type(index) == np.ndarray:
        #     return AroundViewGraspGroup(self.grasp_group_array[index])
        # elif type(index) == list:
        #     return AroundViewGraspGroup(self.grasp_group_array[index])
        else:
            raise TypeError('unknown type "{}" for calling __getitem__ for AroundViewGraspGroup'.format(type(index)))

    @property
    def ann_ids(self):
        return self._ann_ids

    @ann_ids.setter
    def ann_ids(self, ids):
        # assert len(ids) == self.__len__()
        self._ann_ids = ids
    
    def convert_ann_id_matrix(self, original_ann_id, target_ann_id):
        original_camera_pose = self.camera_poses[original_ann_id]
        target_camera_pose = self.camera_poses[target_ann_id]
        return np.matmul(np.linalg.inv(target_camera_pose), original_camera_pose)

    def to_view(self, target_ann_id):
        if Counter(self.ann_ids)[self.ann_ids[0]] == self.__len__():
            # all grasps are in the same ann_id.
            T = self.convert_ann_id_matrix(self.ann_ids[0], target_ann_id)
            self.transform(T)
            self.ann_ids[:] = target_ann_id
        else:
            for i in range(self.__len__()):
                g = self[i].to_view(target_ann_id)
                self.grasp_group_array[i] = g.grasp_array
                # self.ann_ids[i] = g.ann_id
            self.ann_ids[:] = target_ann_id
        return self

    def add(self, element):
        '''
        **Input:**
        - element: AroundViewGrasp instance or AroundViewGraspGroup instance.
        '''
        if isinstance(element, AroundViewGrasp):
            self.ann_ids = np.concatenate((self.ann_ids, element.ann_id.reshape((0))))
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_array.reshape((-1, GRASP_ARRAY_LEN))))
        elif isinstance(element, AroundViewGraspGroup):
            self.ann_ids = np.concatenate((self.ann_ids, element.ann_ids))
            self.grasp_group_array = np.concatenate((self.grasp_group_array, element.grasp_group_array))
        else:
            raise TypeError('Unknown type:{}'.format(element))
        return self

    def sort_by_score(self, reverse=False):
        '''
        **Input:**
        - reverse: bool of order, if False, from high to low, if True, from low to high.

        **Output:**
        - no output but sort the grasp group.
        '''
        score = self.grasp_group_array[:,0]
        index = np.argsort(score)
        if not reverse:
            index = index[::-1]
        self.grasp_group_array = self.grasp_group_array[index]
        self.ann_ids = self.ann_ids[index]
        return self
