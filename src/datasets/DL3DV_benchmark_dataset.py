import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.os_utils import images_names_from_dir
from src.utils.load_utils import load_and_preprocess_images
from src.datasets.datasets_paths import test_dataset_paths

class DL3DVBenchmarkDataset(Dataset):
    def __init__(self, 
                num_source_views=10,
                num_target_views=9,
                loading_mode='vggt_crop',
                load_size=518,
                load_images=True):

        self.dataset_path = test_dataset_paths['DL3DV_benchmark']
        self.num_source_views = num_source_views
        self.num_target_views = num_target_views
        self.load_images = load_images
        self.loading_mode = loading_mode
        self.load_size = load_size
        scene_names = sorted(os.listdir(self.dataset_path))
        try:
            scene_names.remove('.cache')
        except:
            pass
        scene_names.remove('benchmark-meta.csv')
        self.scene_names = scene_names
        
        self.step = 10

    def __len__(self):
        return len(self.scene_names)

    def load_params(self, path):

        with open(path,'r') as f:
            params_dict = json.load(f)

        aspect_ratio = params_dict['w'] / params_dict['h']
        target_h = 518
        target_w = 518 * aspect_ratio

        fx = params_dict['fl_x'] * (target_w/params_dict['w'])
        fy = params_dict['fl_y'] * (target_h/params_dict['h'])
        
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = 518/2
        K[1,2] = 518/2
        return K

    def __getitem__(self, idx):

        scene = self.scene_names[idx]
        
        source_query_frame = 0
        offset = self.step // 2
        target_query_frame = offset
        source_samples_idx = [source_query_frame + n * self.step for n in range(self.num_source_views)]
        target_samples_idx = [target_query_frame + n * self.step for n in range(self.num_target_views)]

        scene_path = os.path.join(self.dataset_path,scene,'nerfstudio/images_4')
        images_names = images_names_from_dir(scene_path)

        transforms_file = os.path.join(self.dataset_path,scene,'nerfstudio/transforms.json')
        K = self.load_params(transforms_file)
        K = np.broadcast_to(K[None],(self.num_source_views,3,3))

        #train_paths = [os.path.join(scene_path,f"frame_{str(sample+1).zfill(5)}.png") for sample in source_samples_idx]
        #test_paths = [os.path.join(scene_path,f"frame_{str(sample+1).zfill(5)}.png") for sample in target_samples_idx]
        train_paths = [os.path.join(scene_path,images_names[sample]) for sample in source_samples_idx]
        test_paths = [os.path.join(scene_path,images_names[sample]) for sample in target_samples_idx]

        if self.load_images:
            train_images = load_and_preprocess_images(train_paths, mode=self.loading_mode, target_size=self.load_size)
            test_images = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=self.load_size)

        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
            #data["depths"] = train_depths
        #data["intrinsics"] = train_intrinsics.numpy()
        #data["test_intrinsics"] = test_intrinsics.numpy()
        #data["extrinsics"] = train_extrinsics
        #data["test_extrinsics"] = test_extrinsics
        data["seq"] = scene
        data["train_paths"] = train_paths
        data["test_paths"] = test_paths
        data["train_ids"] = source_samples_idx
        data["test_ids"] = target_samples_idx
        #data["intrinsics"] = torch.from_numpy(K)
        return data
