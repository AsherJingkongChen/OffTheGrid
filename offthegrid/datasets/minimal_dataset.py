import os
import torch
from torch.utils.data import Dataset
from offthegrid.utils.load_utils import load_and_preprocess_images
from offthegrid.datasets.datasets_paths import test_dataset_paths

def images_names_from_dir(directory, filter_str=None):
    image_names = sorted(os.listdir(directory))
    if filter_str is not None:
        image_names = [i for i in image_names if filter_str in i]
    image_names = [i for i in image_names if ('.jpg' in i or '.JPG' in i or '.png' in i)]
    image_names = [os.path.join(directory, x) for x in image_names]
    return image_names

class MinimalDataset(Dataset):
    def __init__(self,
                 num_images_per_scene=3,
                 loading_mode='vggt_crop',
                 load_size=518):

        self.dataset_path = test_dataset_paths.get('minimal_dataset', '')
        if os.path.exists(self.dataset_path):
            self.scene_names = sorted([d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))])
        else:
            self.scene_names = []
            
        self.num_images_per_scene = num_images_per_scene
        self.loading_mode = loading_mode
        self.load_size = load_size

    def __len__(self):
        return len(self.scene_names)

    def load_intrinsics(self, path, target_size):
        raise NotImplementedError

    def load_camera_poses(self, img_paths):
        raise NotImplementedError
    
    def load_depths(self, paths):
        raise NotImplementedError

    def __getitem__(self, idx):
        scene = self.scene_names[idx]
        scene_dir = os.path.join(self.dataset_path, scene)
        
        all_images = images_names_from_dir(scene_dir)
        
        # Interleaved mapping
        source_idx = [i * 2 for i in range(self.num_images_per_scene)]
        target_idx = [i * 2 + 1 for i in range(self.num_images_per_scene)]
        
        # Filter indices to not exceed total images
        source_idx = [i for i in source_idx if i < len(all_images)]
        target_idx = [i for i in target_idx if i < len(all_images)]
        
        train_paths = [all_images[i] for i in source_idx]
        test_paths = [all_images[i] for i in target_idx]

        train_images = load_and_preprocess_images(train_paths, mode=self.loading_mode, target_size=self.load_size) if train_paths else None
        test_images = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=self.load_size) if test_paths else None
        
        data = {}
        data["images"] = train_images
        data["target_views"] = test_images
        data["target_views_original"] = test_images
        
        data["seq"] = scene
        data["train_paths"] = train_paths
        data["test_paths"] = test_paths
        data["train_ids"] = source_idx
        
        return data
