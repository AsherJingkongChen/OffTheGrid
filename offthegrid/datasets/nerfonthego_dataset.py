import os
import json
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from offthegrid.utils.load_utils import load_and_preprocess_images
from offthegrid.datasets.datasets_paths import test_dataset_paths

class NerfOnTheGo_Dataset(Dataset):
    def __init__(self,
                 num_source_views=3,
                 loading_mode='vggt_crop',
                 load_size=518,
                 load_images=True):

        self.dataset_path = test_dataset_paths['nerfonthego']
        self.results_path = os.path.join(test_dataset_paths['nerfonthego'], 'results', 'GenWildSplat')
        
        # Supported regimes: 3, 6, 9, and 12
        self.num_source_views = num_source_views
        self.loading_mode = loading_mode
        self.load_size = load_size
        self.load_images = load_images
        
        # Specific scenes for RobustNeRF
        self.scene_names = ['corner', 'spot', 'patio', 'patio-high', 'mountain', 'fountain']

    def __len__(self):
        return len(self.scene_names)

    def load_intrinsics(self, path, target_size):
        raise NotImplementedError

    def load_camera_poses(self, img_paths):
        raise NotImplementedError
    
    def load_depths(self, paths):
        raise NotImplementedError

    def _resolve_image_dir(self, json_image_dir):
        """
        Helper to resolve potential mount mismatches.
        Maps paths like '/mnt/data/robustnerf/statue/images_8' 
        to '/nfs/cvn/datasets/robustnerf/statue/images_8'
        """
        if "nerfonthego-undistorted" in json_image_dir:
            relative_dir = json_image_dir.split("nerfonthego-undistorted/")[-1]
            return os.path.join(self.dataset_path, relative_dir)
        print(json_image_dir)
        return json_image_dir

    def __getitem__(self, idx):
        scene = self.scene_names[idx]
        
        # Locate the JSON file for the specific scene and view regime
        json_dir = os.path.join(self.results_path, scene, f"{self.num_source_views}_views")
        
        # Find the JSON file in the directory
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON file found in {json_dir}")
        
        # Assuming one relevant json per directory
        json_path = json_files[0] 
        
        with open(json_path, 'r') as f:
            data_info = json.load(f)

        # Extract data from the JSON
        context_images = data_info['context_images']
        target_images = data_info['target_images']
        source_samples_idx = data_info['context_indices']
        
        # Resolve image directory 
        image_dir = os.path.join(self.dataset_path, scene, 'images')

        # Create full paths
        train_paths = [os.path.join(image_dir, img) for img in context_images]
        test_paths = [os.path.join(image_dir, img) for img in target_images]
        if len(test_paths) > 50:
            test_paths = [test_paths[i] for i in range(50)]

        if self.load_images:
            train_images = load_and_preprocess_images(train_paths, mode=self.loading_mode, target_size=self.load_size)
            test_images = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=self.load_size) 
            test_images448 = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=448) 
        else:
            train_images = None
            test_images = None
            test_images448 = None

        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
            data["target_views_original"] = test_images448
            print(train_images.shape, test_images.shape)
        data["seq"] = scene
        data["train_paths"] = train_paths
        data["test_paths"] = test_paths
        data["train_ids"] = source_samples_idx
        
        return data

if __name__ == "__main__":
    # Testing the 3-view regime
    dataset = RobustNerf_Dataset(num_source_views=3, load_images=False)
    
    for k in range(len(dataset)):
        d = dataset[k]
        print(f"Scene: {d['seq']}")
        print(f"Number of Train Paths: {len(d['train_paths'])}")
        print(f"Number of Test Paths: {len(d['test_paths'])}")
        print("-" * 30)