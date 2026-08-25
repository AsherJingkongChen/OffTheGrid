import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from src.utils.os_utils import images_names_from_dir
from src.utils.load_utils import load_and_preprocess_images
from src.datasets.datasets_paths import test_dataset_paths


tandt_path = test_dataset_paths['tanks_and_temples']
    
class TanksAndTemplesDataset(Dataset):
    def __init__(self, num_source_views=12, num_target_views=11, split='Intermediate',
                 step=2, loading_mode='vggt_crop',
                load_size=518, load_images=True):
        self.dataset_path = tandt_path
        self.num_source_views = num_source_views
        self.num_target_views = num_target_views
        self.load_images = load_images
        self.loading_mode = loading_mode
        self.load_size = load_size
        if split == 'all':
            self.splits = ['Intermediate', 'Advanced', 'Train']
        else:
            self.splits = [split]
        
        scene_names = []
        self.num_intermediate = 8
        self.num_advanced = 6
        
        for split in self.splits:
            if split == 'Train':
                data_path = os.path.join(self.dataset_path,split)
            else:
                data_path = os.path.join(self.dataset_path,'Test',split)
            scene_names += sorted(os.listdir(data_path))
        
        self.dataset_h = 1080
        self.dataset_w = 1920
        self.scene_names = scene_names
        
        if num_source_views<4 :
            self.step = 8
        if num_source_views<10 :
            self.step=4
        else:
            self.step=2
    def __len__(self):
        return len(self.scene_names)


    def __getitem__(self, idx):

        scene = self.scene_names[idx]
        
        source_query_frame = 0
        offset = self.step // 2
        target_query_frame = offset
        source_samples_idx = [source_query_frame + n * self.step for n in range(self.num_source_views)]
        target_samples_idx = [target_query_frame + n * self.step for n in range(self.num_target_views)]

        if idx<self.num_intermediate:
            scene_path = os.path.join(self.dataset_path,'Test',self.splits[0],scene)
        elif idx<self.num_intermediate+self.num_advanced:
            scene_path = os.path.join(self.dataset_path,'Test',self.splits[1],scene)
        else:
            scene_path = os.path.join(self.dataset_path,self.splits[2],scene)

        images_names = images_names_from_dir(scene_path)
        #print(scene, len(images_names))
        train_paths = [os.path.join(scene_path,images_names[sample]) for sample in source_samples_idx]
        test_paths = [os.path.join(scene_path,images_names[sample]) for sample in target_samples_idx]

        if self.load_images:
            train_images = load_and_preprocess_images(train_paths, mode=self.loading_mode, target_size=self.load_size)
            test_images = load_and_preprocess_images(test_paths, mode=self.loading_mode, target_size=self.load_size)

        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
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