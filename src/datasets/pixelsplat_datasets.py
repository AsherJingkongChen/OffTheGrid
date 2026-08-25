import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms as TF
from torch.utils.data import Dataset
from src.utils.os_utils import images_names_from_dir
from src.utils.load_utils import load_and_preprocess_images

class PixelSplatDataset(Dataset):
    def __init__(self, data_path, load_images=True):

        self.dataset_path = data_path
        self.load_images = load_images
        
        scene_names = sorted(os.listdir(self.dataset_path))
        scene_names.remove("peak_memory.json")
        scene_names.remove("benchmark.json")
        self.dataset_h = 256
        self.dataset_w = 256
        self.scene_names = scene_names
        
    def __len__(self):
        return len(self.scene_names)

    def load_256_images(self, paths):
        imgs = []
        to_tensor = TF.ToTensor()
        for path in paths:
            img = Image.open(path)
            img = img.convert("RGB")
            img = to_tensor(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs

    def __getitem__(self, idx):

        scene = self.scene_names[idx]
        source_path = os.path.join(self.dataset_path,scene,'context')
        target_path = os.path.join(self.dataset_path,scene,'gt')
        source_images_names = images_names_from_dir(source_path)
        target_images_names = images_names_from_dir(target_path)
        train_paths = [os.path.join(source_path,im) for im in source_images_names]
        test_paths = [os.path.join(target_path,im) for im in target_images_names]

        if self.load_images:
            train_images = load_and_preprocess_images(train_paths, mode='center_crop', target_size=518)
            test_images_original = load_and_preprocess_images(test_paths, mode='center_crop', target_size=256)
            test_images = load_and_preprocess_images(test_paths, mode='center_crop', target_size=518)
        data = {}
        if self.load_images:
            data["images"] = train_images
            data["target_views"] = test_images
            data["target_views_original"] = test_images_original
        data["seq"] = scene
        data["train_paths"] = train_paths
        data["test_paths"] = test_paths
        return data

if __name__ == "__main__":
    dataset = SCRREAMDataset(10, 9, True)
    for k in range(1):
        d = dataset.__getitem__(0)
        print(d["depths_gt"].shape)