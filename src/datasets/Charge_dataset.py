import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from src.utils.load_utils import load_and_preprocess_images
from src.datasets.datasets_paths import test_dataset_paths

class ChargeDataset(Dataset):
    def __init__(self,
                input_mode='sparse_9',
                num_scenes_per_seq=5,
                loading_mode='vggt_crop',
                load_size=518,
                load_images=True):
        
        self.input_mode = input_mode
        self.dataset_path = test_dataset_paths['charge']
        self.sequences = ['040_0040','010_0050','050_0160','060_0130','020_0020','060_0100','070_0123','050_0130']
        self.num_scenes_per_seq = num_scenes_per_seq
        self.loading_mode = loading_mode
        self.load_size = load_size
        self.dataset_h = 858
        self.dataset_w = 2048
        self.load_images = load_images
        
    def __len__(self):
        return len(self.sequences) * self.num_scenes_per_seq

    def load_depths(self, paths):
        depths = load_and_preprocess_images(paths, self.loading_mode, target_size=self.load_size)
        depths = torch.clamp(depths,min=0, max=20)
        return depths

    def transform_intrinsics(self, intrinsics, target_size=(518,518), patch_size=14):
        
        final_h, final_w = target_size
        aspect_ratio = self.dataset_w / self.dataset_h
        if self.loading_mode == 'center_crop':
            target_h = self.load_size
            target_w = self.load_size * aspect_ratio
        elif self.loading_mode in ['no_crop','vggt_crop','vggt_pad']:
            target_w = self.load_size
            target_h = self.load_size / aspect_ratio
            target_h = round(target_h / patch_size) * patch_size
        elif self.loading_mode == 'resize':
            target_h = self.load_size
            target_w = self.load_size
        else:
            raise NotImplementedError

        fx = intrinsics[:,0,0] * (target_w/self.dataset_w)
        fy = intrinsics[:,1,1] * (target_h/self.dataset_h)    
        K = torch.zeros(intrinsics.shape[0],3,3)
        K[:,2,2] = 1 
        K[:,0,0] = torch.from_numpy(fx)
        K[:,1,1] = torch.from_numpy(fy)
        K[:,0,2] = final_w/2
        K[:,1,2] = final_h/2
        return K

    def load_camera_params(self, split, seq, frame_idx):
        cam_infos = {}
        cam_infos["extrinsics"] = []
        cam_infos["intrinsics"] = []
        cam_infos["images_path"] = []
        #cam_infos["cam_centers"] = []
        if split == 'train':
            if self.input_mode == 'sparse_9':
                file_path = os.path.join(self.dataset_path,seq,'Sparse','transforms_train_9views.json')
            elif self.input_mode == 'sparse_6':
                file_path = os.path.join(self.dataset_path,seq,'Sparse','transforms_train_6views.json')
            elif self.input_mode == 'sparse_3':
                file_path = os.path.join(self.dataset_path,seq,'Sparse','transforms_train_3views.json')
            elif self.input_mode == 'dense':
                file_path = os.path.join(self.dataset_path,seq,'Dense','transforms_train.json')
        else:
            if self.input_mode == 'dense':
                file_path = os.path.join(self.dataset_path,seq,'Dense','transforms_test.json')
            else:
                file_path = os.path.join(self.dataset_path,seq,'Sparse','transforms_test.json')
        with open(file_path) as json_file:
            contents = json.load(json_file)
            if split == 'train':
                if self.input_mode == 'dense':
                    num_frames = len(contents["Dense_00_00"])
                else:
                    num_frames = len(contents['Sparse_Train_00'])
            else:
                if self.input_mode == 'dense':
                    num_frames = len(contents['DenseTest_00_00'])
                else:
                    num_frames = len(contents['Sparse_Test_00'])
            selected_frames = [(k+1) * (num_frames//self.num_scenes_per_seq) - 1 for k in range(self.num_scenes_per_seq)]
            current_frame = selected_frames[frame_idx]
            for cam in contents.keys():
                cam_data = contents[cam][current_frame]
                cam_infos["frame_id"] = cam_data["frame_id"]
                
                matrix = np.linalg.inv(np.array(cam_data["transformation_matrix"]))
                matrix[1:3, :] *= -1
                # print(matrix)
                #R = np.transpose(matrix[:3,:3])
                R = matrix[:3,:3]
                # R[:,0] = -R[:,0]
                T = matrix[:3, 3]
                cam_center = (-R @ T[:,None])[:,0]
                #cam_infos["cam_centers"].append(cam_center)
                cam_infos["extrinsics"].append(matrix)

                K = np.array(cam_data["K"])
                cam_infos["intrinsics"].append(K)

                if self.input_mode == 'dense':
                    im_path = os.path.join(self.dataset_path,seq,'Dense',cam_data["image_path"])
                else:
                    im_path = os.path.join(self.dataset_path,seq,'Sparse',cam_data["image_path"])
                cam_infos["images_path"].append(im_path)

        cam_infos["extrinsics"] = np.stack(cam_infos["extrinsics"],axis=0)
        cam_infos["K"] = np.stack(cam_infos["intrinsics"],axis=0)
        #cam_infos["cam_centers"] = np.stack(cam_infos["cam_centers"],axis=0)
        return cam_infos

    def __getitem__(self, idx):
        seq_idx = idx // self.num_scenes_per_seq
        frame_idx = idx % self.num_scenes_per_seq
        seq = self.sequences[seq_idx]
        
        train_cam_info = self.load_camera_params('train', seq, frame_idx)
        test_cam_info = self.load_camera_params('test', seq, frame_idx)

        frame = train_cam_info["frame_id"]
        
        if self.load_images:
            train_images = load_and_preprocess_images(train_cam_info['images_path'], mode=self.loading_mode, target_size=self.load_size)
            test_images = load_and_preprocess_images(test_cam_info['images_path'], mode=self.loading_mode, target_size=self.load_size)
        
        H, W = train_images.shape[-2:]
        #Rescale intrinsics
        train_K = self.transform_intrinsics(train_cam_info["K"], (H,W))
        test_K = self.transform_intrinsics(test_cam_info["K"], (H,W))

        train_depth_paths = [path[:-4] + '_depth.npy' for path in train_cam_info['images_path']]
        train_depths = self.load_depths(train_depth_paths)

        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
            data["depths_gt"] = train_depths
        data["intrinsics"] = train_K
        data["test_intrinsics"] = test_K
        data["extrinsics"] = torch.from_numpy(train_cam_info["extrinsics"])
        data["test_extrinsics"] = torch.from_numpy(test_cam_info["extrinsics"])
        data["seq"] = seq
        data["train_paths"] = train_cam_info['images_path']
        data["test_paths"] = test_cam_info['images_path']
        return data

def charge_collate_fn(batch):
  return {
      'images': torch.stack([x['images'] for x in batch]),
      'target_views': torch.stack([x['target_views'] for x in batch]),
      'depths_gt': torch.stack([x['depths_gt'] for x in batch]),
      'intrinsics': torch.stack([x['intrinsics'] for x in batch]),
      'test_intrinsics': torch.stack([x['test_intrinsics'] for x in batch]),
      'extrinsics': torch.stack([x['extrinsics'] for x in batch]),
      'test_extrinsics': torch.stack([x['test_extrinsics'] for x in batch]),
      'seq': [x['seq'] for x in batch],
      'train_paths': [x['train_paths'] for x in batch],
      'test_paths': [x['test_paths'] for x in batch],
}



def transform_intrinsics_prev(intrinsics, target_size=(518,518)):

    original_h = 858
    original_w = 2048
    #new_height = target_size
    #new_width = round(original_w * (new_height / original_h))
    new_height, new_width = target_size
    x_ratio = new_width / original_w
    y_ratio = new_height / original_h

    fxs = intrinsics[:,0,0] * x_ratio
    fys = intrinsics[:,1,1] * y_ratio

    new_intrinsics = np.zeros((intrinsics.shape[0],3,3))
    new_intrinsics[:,0,0] = fxs
    new_intrinsics[:,1,1] = fys
    new_intrinsics[:,0,2] = new_width // 2
    new_intrinsics[:,1,2] = new_height // 2
    new_intrinsics[:,2,2] = 1
    return new_intrinsics