from graspnetAPI.grasp import Grasp, GraspGroup


class AroundViewGrasp(Grasp):
    @property
    def ann_id(self):
        return self._ann_id
    
    @ann_id.setter
    def ann_id(self, idx):
        assert idx in range(256)
        self._ann_id = idx

    def to_view(self, target_id):
        T = some_magic(self.ann_id, target_id)
        self.transform(T)
        self.ann_id = target_id


class AroundViewGraspGroup(GraspGroup):
    pass
