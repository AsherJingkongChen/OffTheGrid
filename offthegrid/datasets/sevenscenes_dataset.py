import numpy as np
from torch.utils.data import Dataset
import os
import torch
from offthegrid.utils.load_utils import load_and_preprocess_images
from offthegrid.datasets.datasets_paths import test_dataset_paths
from scipy.spatial.transform import Rotation as R

class SevenScenesDataset(Dataset):
    def __init__(self,
                num_source_views=10,
                num_target_views=9, 
                loading_mode='vggt_crop',
                load_size=518,
                load_images=True):
        
        self.num_source_views = num_source_views
        self.num_target_views = num_target_views
        self.load_images = load_images
        self.dataset_path = test_dataset_paths['sevenscenes']
        self.scenes = ['chess','fire','heads','office','pumpkin','redkitchen','stairs']
        self.test_sequences = [[3,5],[3,4],[1],[2,6,7,9],[1,7],[3,4,6,12,14], [1,4]]
        self.loading_mode = loading_mode
        self.load_size = load_size
        self.step = 15
        self.start_frame = 0
        
        self.camera_poses_path = '../visloc_pseudo_gt_limitations/pgt/sfm/7scenes/' 
        #download here https://github.com/tsattler/visloc_pseudo_gt_limitations

        #self.gt_poses = {}
        #for scene in self.scenes:
        #    self.gt_poses[scene] = self.load_camera_params(scene)

    def __len__(self):
        return len(self.scenes)

    def load_depths(self, paths):
        depths_paths = [a.replace('color','depth') for a in paths]
        depths = load_and_preprocess_images(depths_paths, mode=self.loading_mode, target_size=self.load_size)
        depths = depths[:,:1]
        return depths
        
    def load_camera_params(self, scene):
        file_path = os.path.join(self.camera_poses_path,f"{scene}_test.txt")

        with open(file_path, "r") as f:
            pose_data = f.readlines()

        pose_dict = {}
        for pose_string in pose_data:

            pose_string = pose_string.split()
            file_name = pose_string[0]

            pose_q = np.array(pose_string[1:5])
            pose_q = np.array([pose_q[1], pose_q[2], pose_q[3], pose_q[0]])
            pose_t = np.array(pose_string[5:8])
            pose_R = R.from_quat(pose_q).as_matrix()

            pose_4x4 = np.identity(4)
            pose_4x4[0:3, 0:3] = pose_R
            pose_4x4[0:3, 3] = pose_t

            # convert world->cam to cam->world for evaluation
            #pose_4x4 = np.linalg.inv(pose_4x4)

            if len(pose_string) > 8:
                focal_length = float(pose_string[8])
            else:
                focal_length = None

            pose_dict[file_name] = (pose_4x4, focal_length)
        return pose_dict

    def __getitem__(self, idx):

        scene = self.scenes[idx]
        seq = self.test_sequences[idx][0]

        folder = os.path.join(self.dataset_path,scene,f'seq-0{str(seq)}')
        
        
        train_samples_idx = [self.start_frame + n * self.step for n in range(self.num_source_views)]
        shift = self.step // 2
        test_samples_idx = [self.start_frame + shift + n * self.step for n in range(self.num_target_views)]
        #train_paths = [os.path.join(folder,f'frame-{str(i).zfill(6)}.color.png') for i in train_samples_idx]
        #test_paths = [os.path.join(folder,f'frame-{str(i).zfill(6)}.color.png') for i in test_samples_idx]
        train_paths = [f'frame-{str(i).zfill(6)}.color.png' for i in train_samples_idx]
        test_paths = [f'frame-{str(i).zfill(6)}.color.png' for i in test_samples_idx]

        """train_extrinsics = []
        train_focal = []
        for im_name in train_paths:
            key = f'seq-0{str(seq)}/' + im_name
            extrinsics, focal_length = self.gt_poses[scene][key]
            train_extrinsics.append(extrinsics)
            if focal_length is not None:
                train_focal.append(focal_length)
        focal = np.mean(train_focal)
        focal = focal * (518/480)
        train_extrinsics = np.stack(train_extrinsics,axis=0)
        train_intrinsics = torch.eye(3)
        train_intrinsics[0,2] = 518 /2
        train_intrinsics[1,2] = 518 /2
        train_intrinsics[0,0] = focal
        train_intrinsics[1,1] = focal
        train_intrinsics = train_intrinsics[None].expand(train_extrinsics.shape[0],3,3)

        test_extrinsics = []
        test_focal = []
        for im_name in test_paths:
            key = f'seq-0{str(seq)}/' + im_name
            extrinsics, focal_length = self.gt_poses[scene][key]
            test_extrinsics.append(extrinsics)
            if focal_length is not None:
                test_focal.append(focal_length)
        focal = np.mean(test_focal) * (518/480)
        test_extrinsics = np.stack(test_extrinsics,axis=0)
        test_intrinsics = torch.eye(3)
        test_intrinsics[0,2] = 518 /2
        test_intrinsics[1,2] = 518 /2
        test_intrinsics[0,0] = focal
        test_intrinsics[1,1] = focal
        test_intrinsics = test_intrinsics[None].expand(test_extrinsics.shape[0],3,3)
        """
        train_paths = [os.path.join(folder,train_path) for train_path in train_paths]
        test_paths = [os.path.join(folder,test_path) for test_path in test_paths]

        if self.load_images:
            train_images = load_and_preprocess_images(train_paths, mode=self.loading_mode, target_size=self.load_size)
            test_images = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=self.load_size)
            #train_depths = self.load_depths(train_paths)

        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
            #data["depths"] = train_depths
        #data["intrinsics"] = train_intrinsics.numpy()
        #data["test_intrinsics"] = test_intrinsics.numpy()
        #data["extrinsics"] = train_extrinsics
        #data["test_extrinsics"] = test_extrinsics
        data["seq"] = seq
        data["train_paths"] = train_paths
        data["test_paths"] = test_paths
        return data