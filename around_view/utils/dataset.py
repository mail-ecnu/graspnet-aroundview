import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from datasets.graspnet_dataset import GraspNetDataset


class AroundViewDataset(GraspNetDataset):
    '''
        dataset for View selecting
    '''
    def __init__(self, root, camera, split='train', num_points=20000, augment=False):
        super().__init__(root, None, None, camera='kinect', split='train', 
            num_points=num_points, remove_outlier=True, augment=augment)

    def __getitem__(self, index):
        '''
            label: generate label after forward
        '''
        return self.get_data(index)
