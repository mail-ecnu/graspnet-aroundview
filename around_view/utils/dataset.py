import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.io as scio

import torch
from torch._six import container_abcs

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models.graspnet import GraspNet
from dataset.graspnet_dataset import GraspNetDataset
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from around_view.models.pytorch_utils import to_var

VIEW_LEN = 16
ALL_ANN_IDs = np.array([x*(256 // VIEW_LEN) for x in range(VIEW_LEN)])


class AroundViewDataset(GraspNetDataset):
    '''
        dataset for View selecting
    '''
    def __init__(self, root, camera, split='train', num_points=20000, augment=False):
        super().__init__(root, None, None, camera, split=split, 
            num_points=num_points, remove_outlier=True, augment=augment, load_label=False)

        all_len = len(self.colorpath)
        self.colorpath = [self.colorpath[i: i+256] for i in range(0, all_len, 256)]
        self.depthpath = [self.depthpath[i: i+256] for i in range(0, all_len, 256)]
        self.labelpath = [self.labelpath[i: i+256] for i in range(0, all_len, 256)]
        self.metapath = [self.metapath[i: i+256] for i in range(0, all_len, 256)]
        self.scenename = [self.scenename[i: i+256] for i in range(0, all_len, 256)]
        self.frameid = [self.frameid[i: i+256] for i in range(0, all_len, 256)]

        self.colorpath = np.array([np.array(x)[ALL_ANN_IDs] for x in self.colorpath])
        self.depthpath = np.array([np.array(x)[ALL_ANN_IDs] for x in self.depthpath])
        self.labelpath = np.array([np.array(x)[ALL_ANN_IDs] for x in self.labelpath])
        self.metapath = np.array([np.array(x)[ALL_ANN_IDs] for x in self.metapath])
        self.scenename = np.array([np.array(x)[ALL_ANN_IDs] for x in self.scenename])
        self.frameid = np.array([np.array(x)[ALL_ANN_IDs] for x in self.frameid])

    def _get_one_annid_data(self, scene_index, view_index,return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[scene_index][view_index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[scene_index][view_index]))
        seg = np.array(Image.open(self.labelpath[scene_index][view_index]))
        meta = scio.loadmat(self.metapath[scene_index][view_index])
        scene = self.scenename[scene_index][view_index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[scene_index][view_index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        ret_dict = dict(
            scene_indexs=scene_index,
            point_clouds=cloud_sampled.astype(np.float32),
            cloud_colors=color_sampled.astype(np.float32),
        )
        return ret_dict

    def __getitem__(self, index):
        '''label: generate label after forward
        '''
        one_scene_data = []
        for view_id in range(VIEW_LEN):
            one_scene_data.append(self._get_one_annid_data(scene_index=index, view_index=view_id))
        return one_scene_data


def collate_fn_av(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0][0], container_abcs.Mapping):
        return dict(
            scene_indexs=collate_fn([[d['scene_indexs'] for d in one_data] for one_data in batch]),
            point_clouds=collate_fn([[d['point_clouds'] for d in one_data] for one_data in batch]),
        )
    elif isinstance(batch[0], container_abcs.Sequence):
        return torch.from_numpy(np.array(batch))

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def get_backbone(camera):
    full_net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    full_net.to(device)

    # Load checkpoint
    checkpoint_path = {
        'kinect': '/mnt/ssd/chenweiwen/graspnet-baseline/logs/log_kn/checkpoint.tar',
        'realsense': '/mnt/ssd/chenweiwen/graspnet-baseline/logs/log_rs/checkpoint.tar',
    }[camera]
    checkpoint = torch.load(checkpoint_path)
    full_net.load_state_dict(checkpoint['model_state_dict'])
    net = full_net.view_estimator.backbone
    return net


class AroundViewFeatureDataset(AroundViewDataset):
    def __init__(self, root, camera, split='train', num_points=20000, augment=False):
        super().__init__(root, camera, split=split, num_points=20000, augment=False)
        self.generate_feature(root, camera)
        

    def generate_feature(self, root, camera):
        self.featspath = list()
        net = get_backbone(camera)

        for scene_id in range(len(self.sceneIds)):
            feats_dir = os.path.join(root, 'scenes', self.sceneIds[scene_id], camera, 'feature')
            if not os.path.exists(feats_dir):
                os.mkdir(feats_dir)

            one_scene_feats = list()
            for view_id in range(VIEW_LEN):
                feat_path = self.depthpath[scene_id][view_id].replace('depth', 'feature').replace('.png', '.npy')
                if not os.path.exists(feat_path):
                    point_clouds = self._get_one_annid_data(scene_index=scene_id, view_index=view_id)['point_clouds']
                    point_feature, _, _ = net(to_var(point_clouds).unsqueeze(0))
                    point_feature = point_feature.squeeze().cpu().detach().numpy()
                    np.save(feat_path, point_feature)
                one_scene_feats.append(feat_path)
            self.featspath.append(one_scene_feats)

    def __getitem__(self, index):
        one_scene_data = []
        for view_id in range(VIEW_LEN):
            one_scene_data.append(dict(
                scene_index=index,
                cloud_feats=np.load(self.featspath[index][view_id])
            ))
        return one_scene_data


def collate_fn_avf(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0][0], container_abcs.Mapping):
        return dict(
            scene_index=collate_fn_avf([[d['scene_index'] for d in one_data] for one_data in batch]),
            cloud_feats=collate_fn_avf([[d['cloud_feats'] for d in one_data] for one_data in batch]),
        )
    elif isinstance(batch[0], container_abcs.Sequence):
        return torch.from_numpy(np.array(batch))
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
