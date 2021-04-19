import os
import numpy as np

from graspnetAPI import GraspNetEval
from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.utils import generate_scene_model
from graspnetAPI.utils.eval_utils import get_scene_name, create_table_points, voxel_sample_points, transform_points, eval_grasp

from around_view.utils.grasp import AroundViewGrasp, AroundViewGraspGroup


class AroundViewGraspEval(GraspNetEval):
    def __init__(self, root, camera, split='test', method='random'):
        super(AroundViewGraspEval, self).__init__(root, camera, split)
        self.method = method

    def unparallel_eval_scenes(self, scene_ids, dump_folder, proc = 2):
        '''
        not parallel, just for debug, commit it when coding finished
        '''
        res_list = []
        for scene_id in scene_ids:
            res_list.append(self.eval_scene(scene_id, dump_folder))
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_scene(self, scene_id, dump_folder, TOP_K = 50, return_list = False,vis = False, max_width = 0.1):
        '''
        **Input:**
        - scene_id: int of the scene index.
        - dump_folder: string of the folder that saves the dumped npy files.
        - TOP_K: int of the top number of grasp to evaluate
        - return_list: bool of whether to return the result list.
        - vis: bool of whether to show the result
        - max_width: float of the maximum gripper width in evaluation

        **Output:**
        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        dump_dir = os.path.join(dump_folder, get_scene_name(scene_id), self.camera)
        views = np.load(os.path.join(dump_dir, f'{self.method}_views.npy'))
        camera_poses_path = os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera, 'camera_poses.npy')
        grasp_group = AroundViewGraspGroup().from_npy(os.path.join(dump_dir, f'{self.method}_'), camera_poses_path)

        for ann_id in np.unique(grasp_group.ann_ids):
            sub_grasp_group = grasp_group[np.argwhere(grasp_group.ann_ids==ann_id).flatten()]
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = sub_grasp_group.grasp_group_array
            min_width_mask = (gg_array[:,1] < 0)
            max_width_mask = (gg_array[:,1] > max_width)
            gg_array[min_width_mask,1] = 0
            gg_array[max_width_mask,1] = max_width
            sub_grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(sub_grasp_group, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008, TOP_K = TOP_K)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            
            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id , ann_id, return_poses=False, align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

        grasp_list = np.concatenate(grasp_list_list)
        score_list = np.concatenate(score_list_list)
        collision_mask_list = np.concatenate(collision_list_list)

        # sort in scene level
        grasp_confidence = grasp_list[:,0]
        indices = np.argsort(-grasp_confidence)
        grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

        #calculate AP
        grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
        for fric_idx, fric in enumerate(list_coe_of_friction):
            for k in range(0,TOP_K):
                if k+1 > len(score_list):
                    grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                else:
                    grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

        # print('\rMean Accuracy for scene:%04d = %.3f' % (scene_id, 100.0 * np.mean(grasp_accuracy[:,:])))
        print('\rMean Accuracy for scene:%04d = %.3f' % (scene_id, 100.0 * np.mean(grasp_accuracy[:,:])), end='', flush=True)
        scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list


class AroundViewGraspEval_TransMat(GraspNetEval):
# class AroundViewGraspEval(GraspNetEval):
    '''
        PlanA for AroundView, which has csl's transMat
        but seems a little bug
    '''
    def __init__(self, root, camera, split='test', method='random'):
        super(AroundViewGraspEval, self).__init__(root, camera, split)
        self.method = method

    def parallel_eval_scenes(self, scene_ids, dump_folder, proc = 2):
        '''
        not parallel, just for debug, commit it when coding finished
        '''
        res_list = []
        for scene_id in scene_ids:
            res_list.append(self.eval_scene(scene_id, dump_folder))
        scene_acc_list = []
        for res in res_list:
            scene_acc_list.append(res.get())
        return scene_acc_list

    def eval_scene(self, scene_id, dump_folder, TOP_K = 50, return_list = False,vis = False, max_width = 0.1):
        '''
        **Input:**
        - scene_id: int of the scene index.
        - dump_folder: string of the folder that saves the dumped npy files.
        - TOP_K: int of the top number of grasp to evaluate
        - return_list: bool of whether to return the result list.
        - vis: bool of whether to show the result
        - max_width: float of the maximum gripper width in evaluation

        **Output:**
        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        
        list_coe_of_friction = [0.2,0.4,0.6,0.8,1.0,1.2]

        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        # for ann_id in range(256):
        dump_dir = os.path.join(dump_folder, get_scene_name(scene_id), self.camera)
        views = np.load(os.path.join(dump_dir, f'{self.method}_views.npy'))
        ann_id = views[0]
        grasp_group = GraspGroup().from_npy(os.path.join(dump_dir, f'{self.method}_grasps.npy'))
        _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
        table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

        # clip width to [0,max_width]
        gg_array = grasp_group.grasp_group_array
        min_width_mask = (gg_array[:,1] < 0)
        max_width_mask = (gg_array[:,1] > max_width)
        gg_array[min_width_mask,1] = 0
        gg_array[max_width_mask,1] = max_width
        grasp_group.grasp_group_array = gg_array

        grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list, pose_list, config, table=table_trans, voxel_size=0.008, TOP_K = TOP_K)

        # remove empty
        grasp_list = [x for x in grasp_list if len(x) != 0]
        score_list = [x for x in score_list if len(x) != 0]
        collision_mask_list = [x for x in collision_mask_list if len(x)!=0]

        if len(grasp_list) == 0:
            grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
            scene_accuracy.append(grasp_accuracy)
            grasp_list_list.append([])
            score_list_list.append([])
            collision_list_list.append([])
            print('\rMean Accuracy for scene:{} ='.format(scene_id),np.mean(grasp_accuracy[:,:]), end='')
        else:
            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            
            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id , ann_id, return_poses=False, align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5 # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype = np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])
            
            # sort in scene level
            grasp_confidence = grasp_list[:,0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            #calculate AP
            grasp_accuracy = np.zeros((TOP_K,len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0,TOP_K):
                    if k+1 > len(score_list):
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                    else:
                        grasp_accuracy[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

            print('\rMean Accuracy for scene:%04d = %.3f' % (scene_id, 100.0 * np.mean(grasp_accuracy[:,:])), end='', flush=True)
            # import ipdb; ipdb.set_trace()
            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list
