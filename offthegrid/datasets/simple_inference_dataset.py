
import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from offthegrid.utils.load_utils import load_and_preprocess_images, indices_to_select
from offthegrid.vggt.utils.pose_enc import pose_encoding_to_extri_intri

from random import shuffle
from tqdm import tqdm
from torchvision.io import read_image
import time
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def images_names_from_dir(directory, filter_str=None):
    image_names = sorted(os.listdir(directory))
    if filter_str is not None:
        image_names = [i for i in image_names if filter_str in i]
    image_names = [i for i in image_names if ('.jpg' in i or '.JPG' in i or '.png' in i)]
    image_names = [os.path.join(directory, x) for x in image_names]
    return image_names

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, num_images_per_scene,
                 sampling_mode='linear', loading_mode='vggt_crop',
                load_size=518):
        
        self.folder_path = folder_path
        self.num_images_per_scene = num_images_per_scene
        self.sampling_mode = sampling_mode       
        self.length = 1
        self.loading_mode = loading_mode
        self.load_size = load_size

    def sample_images(self, num_images, num_images_total, sample_mode, deterministic=False):
        
        if sample_mode == 'randperm':
            if deterministic:
                samples_idx = torch.arange(num_images_total)[:num_images].tolist()
            else:
                samples_idx = torch.randperm(num_images_total)[:num_images].tolist()
        elif sample_mode == 'close_in_time':
            if deterministic:
                query_frame = num_images_total // 2
                samples = [query_frame + i for i in range(num_images-1)]
            else:
                query_frame = torch.randint(num_images_total, size=(1,))[0].item()
                acceptable_range = 10
                min_sample = max(0,query_frame - acceptable_range)
                max_sample = min(num_images_total,query_frame + acceptable_range + 1)
                samples = [t for t in range(min_sample,max_sample) if t != query_frame]
                shuffle(samples)
            samples_idx = [query_frame] + samples[:num_images-1]
        elif sample_mode == 'linear':
            samples_idx = indices_to_select(num_images_total,num_images)
        elif sample_mode == 'linear_close':
            step = 20
            while num_images * step > num_images_total:
                step -= 1
            if deterministic or num_images_total == num_images * step:
                query_frame = 0
            else:
                query_frame = torch.randint(num_images_total - num_images * step, size=(1,))[0].item()
            samples_idx = [query_frame + n * step for n in range(num_images)]
            if not deterministic:
                shuffle(samples_idx)
        elif sample_mode == 'consecutive':
            start_fid = torch.randint(num_images_total - num_images, size=(1,))[0].item()
            samples_idx = [start_fid + i for i in range(num_images)]
        elif sample_mode == 'consecutive_deterministic':
            start_fid = 0
            samples_idx = [start_fid + i for i in range(num_images)]
        elif sample_mode == 'DL3DV_bench_source':
            step = 5
            query_frame = 0
            samples_idx = [query_frame + n * step for n in range(num_images)]
        elif sample_mode == 'DL3DV_bench_test':
            step = 5
            query_frame = 2
            samples_idx = [query_frame + n * step for n in range(num_images)]
        return samples_idx
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        

        image_names = images_names_from_dir(self.folder_path)
        num_images_total = len(image_names)
        if num_images_total < self.num_images_per_scene:
            print(f"Found {num_images_total} images in folder which is less than specified number of images {self.num_images_per_scene}")
            #self.num_images_per_scene = num_images_total
        num_samples = min(num_images_total,self.num_images_per_scene)
        samples_idx = self.sample_images(num_samples, num_images_total, self.sampling_mode, deterministic=True)
        
        image_samples = [image_names[i] for i in samples_idx]
        images = load_and_preprocess_images(image_samples, mode=self.loading_mode, target_size=self.load_size)
        data = {}
        data["images"] = images[None]
        H, W = data["images"].shape[-2:]
        data["paths"] = image_samples
        #data["im_sizes"] = H,W
    
        return data
    
if __name__ == "__main__":
    dataset = ImageFolderDataset("DL3DV_2k", 10, sampling_mode='consecutive', processing_mode="read_from_disk",use_features=True, use_depth_conf=True,max_samples=1)
    d = dataset.__getitem__(0)
