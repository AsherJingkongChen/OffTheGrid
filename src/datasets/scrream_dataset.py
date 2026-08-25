import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from src.utils.os_utils import images_names_from_dir
from src.utils.load_utils import load_and_preprocess_images
from src.datasets.datasets_paths import test_dataset_paths

class SCRREAMDataset(Dataset):
    def __init__(self,
                num_source_views=10,
                num_target_views=9, 
                loading_mode='vggt_crop',
                load_size=518,
                load_images=True):

        self.dataset_path = test_dataset_paths['scrream']
        self.num_source_views = num_source_views
        self.num_target_views = num_target_views
        self.loading_mode = loading_mode
        self.load_size = load_size
        self.load_images = load_images
        scene_names = sorted(os.listdir(self.dataset_path))
        
        self.dataset_h = 874
        self.dataset_w = 1132
        self.scene_names = scene_names
        if num_source_views < 10:
            self.step = 20
        else:
            self.step = 10

    def __len__(self):
        return len(self.scene_names)

    def load_intrinsics(self, path, target_size):

        final_h, final_w = target_size
        
        intrinsics = np.loadtxt(path)

        aspect_ratio = self.dataset_w / self.dataset_h
        if self.loading_mode == 'center_crop':
            target_h = self.load_size
            target_w = self.load_size * aspect_ratio
        elif self.loading_mode in ['vggt_crop','vggt_pad']:
            patch_size = 14
            target_w = self.load_size
            target_h = self.load_size / aspect_ratio
            target_h = round(target_h / patch_size) * patch_size
        elif self.loading_mode == 'resize':
            target_h = self.load_size
            target_w = self.load_size
        elif self.loading_mode == 'no_crop':
            patch_size = 14
            if self.dataset_w  >= self.dataset_h:
                target_w = self.load_size
                target_h = self.load_size / aspect_ratio
                target_h = round(target_h / patch_size) * patch_size
            else:
                target_h = self.load_size
                target_w = self.load_size / aspect_ratio
                target_w = round(target_w / patch_size) * patch_size
        else:
            raise NotImplementedError

        fx = intrinsics[0,0] * (target_w/self.dataset_w)
        fy = intrinsics[1,1] * (target_h/self.dataset_h)
        
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = final_w/2
        K[1,2] = final_h/2
        K = np.broadcast_to(K[None],(self.num_source_views,3,3))
        K = torch.from_numpy(K.copy())
        return K

    def load_camera_poses(self, img_paths):

        poses_path = [a.replace("rgb","camera_pose").replace("png","txt") for a in img_paths]
        
        cam_poses = []
        for path in poses_path:
            cam_pose = torch.from_numpy(np.loadtxt(path))
            cam_poses.append(cam_pose)
        cam_poses = torch.stack(cam_poses,dim=0)
        cam_poses = torch.inverse(cam_poses)
        return cam_poses[:,:3]
    
    def load_depths(self, paths):
        depths_paths = [a.replace("rgb","depth_gt") for a in paths]
        depths = load_and_preprocess_images(depths_paths, mode=self.loading_mode, target_size=self.load_size) 
        #depths = depths[:,:1] / 1000
        return depths

    def load_depths_prev(self, paths):
        depths_paths = [a.replace("rgb","depth_gt") for a in paths]
        depths = []
        target_size = 518
        for path in depths_paths:
            depth_map = cv2.imread(path,-1) / 1000
            #print(depth_map.shape, np.min(depth_map), np.mean(depth_map), np.max(depth_map))
            #depth_map = np.clip(depth_map,a_min=0, a_max=20)
            height, width = depth_map.shape

            if width <= height:
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width)) 
            else:
                new_height = target_size
                new_width = round(width * (new_height / height))
            
            resize_param = (width / new_width, height / new_height, width, height)
            depth_map = cv2.resize(depth_map[:,:,None], (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                depth_map = depth_map[start_y : start_y + target_size, :]
            if new_width > target_size:
                start_x = (new_width - target_size) // 2
                depth_map = depth_map[:, start_x : start_x + target_size]
            
            depths.append(depth_map.copy())
        return depths


    def __getitem__(self, idx):

        scene = self.scene_names[idx]
        
        source_query_frame = 0
        offset = self.step // 2
        target_query_frame = offset
        source_samples_idx = [source_query_frame + n * self.step for n in range(self.num_source_views)]
        target_samples_idx = [target_query_frame + n * self.step for n in range(self.num_target_views)]

        scene_path = os.path.join(self.dataset_path,scene,f'{scene}_full_00','rgb')
        images_names = images_names_from_dir(scene_path)

        
        
        
        train_paths = [os.path.join(scene_path,images_names[sample]) for sample in source_samples_idx]
        test_paths = [os.path.join(scene_path,images_names[sample]) for sample in target_samples_idx]

        if self.load_images:
            train_images = load_and_preprocess_images(train_paths, mode=self.loading_mode, target_size=self.load_size)
            test_images = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=self.load_size) 

        H, W = train_images.shape[-2:]
        intrinsics_file = os.path.join(self.dataset_path,scene,f'{scene}_full_00','intrinsics.txt')
        
        K = self.load_intrinsics(intrinsics_file, (H,W))

        source_extrinsics = self.load_camera_poses(train_paths)
        target_extrinsics = self.load_camera_poses(test_paths)

        source_depths = self.load_depths(train_paths)

        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
        data["depths_gt"] = source_depths
        data["intrinsics"] = K
        data["test_intrinsics"] = K[:self.num_target_views]
        data["extrinsics"] = source_extrinsics
        data["test_extrinsics"] = target_extrinsics
        data["seq"] = scene
        data["train_paths"] = train_paths
        data["test_paths"] = test_paths
        data["train_ids"] = source_samples_idx
        data["test_ids"] = target_samples_idx
        return data

if __name__ == "__main__":
    dataset = SCRREAMDataset(10, 9, True)
    for k in range(1):
        d = dataset.__getitem__(0)
        print(d["depths_gt"].shape)