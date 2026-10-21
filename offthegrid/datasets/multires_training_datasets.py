import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, BatchSampler, DataLoader
from PIL import Image
from tqdm import tqdm
from random import shuffle
import concurrent.futures
from offthegrid.utils.load_utils import load_and_preprocess_images, get_target_dims
from offthegrid.utils.os_utils import images_names_from_dir
from offthegrid.datasets.datasets_paths import train_dataset_paths, test_dataset_paths

class TrainingDataset(Dataset):
    def __init__(self, 
                 name, 
                 scan_fn, 
                 image_num_range=(1, 24),
                 num_target=0, 
                 target_mode="interpolation",
                 sampling_mode='linear_close',
                 loading_mode='vggt_crop',
                 load_size=518,
                 step_range=(5, 10), 
                 sampling_priority=1.,
                 cache_dir='./dataset_cache',
                 consistency_check=False):
        
        self.name = name
        self.image_num_range = image_num_range
        self.num_target = num_target
        self.target_mode = target_mode
        self.sampling_mode = sampling_mode
        self.loading_mode = loading_mode
        self.load_size = load_size
        self.step_range = step_range
        self.sampling_priority = sampling_priority

        os.makedirs(cache_dir, exist_ok=True)
        
        self.valid_cache_path = os.path.join(cache_dir, f"{name}_{loading_mode}_{load_size}_valid.json")
        self.discard_cache_path = os.path.join(cache_dir, f"{name}_discarded_consistency.json")

        if os.path.exists(self.valid_cache_path):
            print(f"[{name}] Loading cached index from {self.valid_cache_path}...")
            with open(self.valid_cache_path, 'r') as f:
                self.scenes = json.load(f)
        else:
            print(f"[{name}] Cache not found. Scanning and checking consistency...")
            raw_scenes = scan_fn() 
            self.scenes = self._index_and_validate_scenes(raw_scenes, check_consistency=consistency_check)
            
            with open(self.valid_cache_path, 'w') as f:
                json.dump(self.scenes, f)
            print(f"[{name}] Saved index with {len(self.scenes)} valid scenes.")

        # --- Build Resolution Map & Max Lengths ---
        self.res_map = {} 
        self.bucket_max_len = {} # (h,w) -> max_frames
        self.max_scene_len = 0   # Dataset global max
        
        for idx, scene in enumerate(self.scenes):
            h, w = tuple(scene['resolution'])
            n_frames = len(scene['images'])
            
            if (h, w) not in self.res_map:
                self.res_map[(h, w)] = []
                self.bucket_max_len[(h, w)] = 0
            
            self.res_map[(h, w)].append(idx)
            
            # Update max lengths
            if n_frames > self.bucket_max_len[(h, w)]:
                self.bucket_max_len[(h, w)] = n_frames
            if n_frames > self.max_scene_len:
                self.max_scene_len = n_frames

    def _index_and_validate_scenes(self, raw_scenes, check_consistency=False):
        valid_scenes = []
        discarded_scenes = []
        
        for images_list in tqdm(raw_scenes, desc=f"Validating {self.name}"):
            if not images_list: continue
            first_img_path = images_list[0]
            try:
                with Image.open(first_img_path) as img:
                    w_ref, h_ref = img.size
                
                if check_consistency:
                    # (Keep your consistency check logic here)
                    pass 

                if self.load_size % 14 == 0:
                    patch_size = 14
                else:
                    patch_size = 16

                new_w, new_h = get_target_dims(w_ref, h_ref, self.loading_mode, self.load_size, patch_size=patch_size)
                if self.loading_mode == 'vggt_crop':
                    new_h = min(self.load_size, new_h)

                valid_scenes.append({
                    "images": images_list,
                    "resolution": (new_h, new_w) 
                })
            except Exception as e:
                discarded_scenes.append(images_list[0])
                continue

        if discarded_scenes:
            print(f"[{self.name}] Discarded {len(discarded_scenes)} scenes.")
            with open(self.discard_cache_path, 'w') as f:
                json.dump(discarded_scenes, f, indent=2)     
        return valid_scenes

    def __len__(self):
        return len(self.scenes)

    def _get_replacement_idx(self, current_res, req_frames):
        """Pick a random scene from the same bucket that satisfies length."""
        h, w = current_res
        candidates = self.res_map.get((h, w), [])
        
        # In a perfect world, we'd filter candidates by length here, 
        # but that might be slow if the bucket is huge. 
        # We rely on probabilistic resampling since the bucket IS valid.
        if not candidates:
            return np.random.randint(len(self))
        return np.random.choice(candidates)

    def __getitem__(self, idx):
        # Default behavior (fallback)
        current_num_source = self.image_num_range[0] # Lowest default
        
        if isinstance(idx, tuple):
            index, n_src, _ = idx
            current_num_source = n_src
            scene_data = self.scenes[index]
        else:
            index = idx
            scene_data = self.scenes[index]

        image_names = scene_data['images']
        current_res = tuple(scene_data['resolution'])
        
        num_images_total = len(image_names)
        num_to_load = current_num_source + self.num_target

        if num_images_total < num_to_load:
            # Resample from SAME resolution bucket
            new_idx = self._get_replacement_idx(current_res, num_to_load)
            return self.__getitem__((new_idx, current_num_source, 0))

        source_idx, target_idx = sample_targets(
            current_num_source, self.num_target, num_images_total, 
            self.target_mode, self.sampling_mode,
            step_range=self.step_range 
        )
        
        source_paths = [image_names[i] for i in source_idx]
        target_paths = [image_names[i] for i in target_idx]

        try:
            source_images = load_and_preprocess_images(
                source_paths, mode=self.loading_mode, target_size=self.load_size
            )
            
            # Runtime Shape Guard
            _, _, h_loaded, w_loaded = source_images.shape
            expected_h, expected_w = current_res
            if h_loaded != expected_h or w_loaded != expected_w:
                raise ValueError("Shape mismatch")

            data = {"images": source_images}
            
            if target_paths:
                target_images = load_and_preprocess_images(
                    target_paths, mode=self.loading_mode, target_size=self.load_size
                )
                _, _, h_tgt, w_tgt = target_images.shape
                if h_tgt != expected_h or w_tgt != expected_w:
                        raise ValueError("Target Shape mismatch")
                data["target_views"] = target_images
                
            return data

        except Exception as e:
            # Resample from SAME resolution bucket
            print(f"Loading failed on scene {source_paths[0]}. Reload another scene.")
            new_idx = self._get_replacement_idx(current_res, num_to_load)
            return self.__getitem__((new_idx, current_num_source, 0))

class MultiTrainingDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.datasets_names = [d.name for d in datasets]
        self.global_map = [] 
        self.offsets = [] 
        
        print("\n--- Initializing MultiTrainingDataset ---")
        
        current_offset = 0
        for ds_idx, ds in enumerate(datasets):
            self.offsets.append(current_offset)
            
            print(f"\nDataset: {ds.name} (Total: {len(ds)} scenes)")
            print(f"Sampling priority: {ds.sampling_priority}")
            sorted_res = sorted(ds.res_map.items(), key=lambda x: len(x[1]), reverse=True)
            for (h, w), indices in sorted_res:
                print(f"  Shape ({h}, {w}): {len(indices)} scenes")

            for local_idx in range(len(ds)):
                self.global_map.append((ds_idx, local_idx))
            
            current_offset += len(ds)
        
        print(f"\nTotal Aggregate Scenes: {len(self.global_map)}")

    def get_datasets_probabilities(self):
        probs = []
        for dataset in self.datasets:
            probs.append(dataset.sampling_priority)
        probs = np.array(probs)
        probs /= np.sum(probs)
        return probs

    def __len__(self):
        return len(self.global_map)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            global_idx, num_source = idx[0], idx[1] 
            ds_idx, local_idx = self.global_map[global_idx]
            return self.datasets[ds_idx].__getitem__((local_idx, num_source, ds_idx))
        else:
            ds_idx, local_idx = self.global_map[idx]
            return self.datasets[ds_idx].__getitem__(local_idx)

class DynamicBatchSampler(BatchSampler):
    def __init__(self, multi_dataset, max_pixels_per_gpu=48*518*518, 
                 image_num_range=(4, 12), num_gpus=1):
        
        self.multi_dataset = multi_dataset
        self.image_num_range = image_num_range
        self.max_pixels = max_pixels_per_gpu
        self.num_gpus = num_gpus
        
        # Assuming all datasets have same target requirement for now, 
        # or we take the max from the first one as they are usually symmetric.
        self.num_target = multi_dataset.datasets[0].num_target
        
        self.dataset_probs = multi_dataset.get_datasets_probabilities()
        self.dataset_len = len(multi_dataset)

    def __iter__(self):
        # Create pools [ds_idx][res_key]
        pools = []
        for ds in self.multi_dataset.datasets:
            ds_pools = {r: torch.randperm(len(indices)).tolist() for r, indices in ds.res_map.items()}
            pools.append(ds_pools)
        
        while True:
            # 1. Determine Sequence Length FIRST
            n_frames = np.random.randint(self.image_num_range[0], self.image_num_range[1] + 1)
            total_req = n_frames + self.num_target

            # 2. Sample Dataset (With Filtering)
            # Loop until we find a dataset that CAN support this length
            ds_idx = -1
            while True:
                candidate_idx = np.random.choice(len(self.multi_dataset.datasets), p=self.dataset_probs)
                # Quick check: Global max len of dataset
                if self.multi_dataset.datasets[candidate_idx].max_scene_len >= total_req:
                    ds_idx = candidate_idx
                    break
            
            # 3. Sample Resolution (With Filtering)
            ds = self.multi_dataset.datasets[ds_idx]
            
            # Filter buckets: valid if bucket_max_len >= total_req
            valid_keys = [
                k for k in ds.res_map.keys() 
                if ds.bucket_max_len[k] >= total_req
            ]
            
            if not valid_keys:
                # Should not happen if max_scene_len check passed, but race condition safety:
                continue

            # Weight by number of scenes in bucket (standard logic)
            counts = [len(ds.res_map[k]) for k in valid_keys]
            probs = np.array(counts, dtype=np.float32) / sum(counts)
            
            res_idx = np.random.choice(len(valid_keys), p=probs)
            res_key = valid_keys[res_idx]
            h, w = res_key
            
            # 4. Batch Specs
            pixels_per_seq = n_frames * h * w
            batch_size = max(1, self.max_pixels // pixels_per_seq)
            if batch_size >= self.num_gpus:
                batch_size = (batch_size // self.num_gpus) * self.num_gpus
            else:
                batch_size = self.num_gpus 
            
            # 5. Fetch Indices
            pool = pools[ds_idx][res_key]
            
            while len(pool) < batch_size:
                # Refill
                new_indices = torch.randperm(len(ds.res_map[res_key])).tolist()
                pool.extend(new_indices)

            batch_local_indices = [pool.pop() for _ in range(batch_size)]
            
            # 6. Global Indices
            raw_indices = ds.res_map[res_key]
            batch_global_indices = []
            for ptr in batch_local_indices:
                local_scene_idx = raw_indices[ptr]
                global_scene_idx = self.multi_dataset.offsets[ds_idx] + local_scene_idx
                batch_global_indices.append(global_scene_idx)
            
            yield [(idx, n_frames, 0) for idx in batch_global_indices]

    def __len__(self):
        avg_img_num = sum(self.image_num_range) / 2
        avg_batch = 4 
        return self.dataset_len // avg_batch
        
# --- Sampling Utils (Updated) ---

def sample_targets(num_source, num_target, num_images_total, target_mode, sample_mode, 
                   deterministic=False, step_range=(5, 10)):
    
    samples_idx = sample_images(num_source+num_target, num_images_total, sample_mode, 
                                deterministic, step_range)
    
    if num_target == 0:
        return samples_idx, []
    
    if target_mode=='interpolation':
        if num_source>1:
            s = list(range(1,len(samples_idx)-1))
        else:
            s = list(range(len(samples_idx)))
        shuffle(s)
        targets = s[:num_target]
        target_samples_idx = [samples_idx[a] for a in targets]
        source_samples_idx = [samples_idx[a] for a in range(len(samples_idx)) if a not in targets]
    elif target_mode == 'future_linear':
        source_samples_idx = samples_idx[:num_source]
        target_samples_idx = samples_idx[num_source:]
    elif target_mode == 'random':
        shuffle(samples_idx)
        source_samples_idx = samples_idx[:num_source]
        target_samples_idx = samples_idx[num_source:]
    
    if not deterministic:
        shuffle(source_samples_idx)
    return source_samples_idx, target_samples_idx

def sample_images(num_images, num_images_total, sample_mode, deterministic=False, step_range=(5, 10)):
    if sample_mode == 'randperm':
        if deterministic:
            samples_idx = torch.arange(num_images_total)[:num_images].tolist()
        else:
            samples_idx = torch.randperm(num_images_total)[:num_images].tolist()
    elif sample_mode == 'linear_close':
        min_step, max_step = step_range
        if deterministic:
            step = min_step
        else:
            step = np.random.randint(min_step, max_step + 1)
            
        while num_images * step > num_images_total and step > 1:
            step -= 1
        max_start = max(0, num_images_total - num_images * step)
        start = 0 if deterministic else np.random.randint(0, max_start + 1)
        samples_idx = [start + i * step for i in range(num_images)]
    else:
        samples_idx = torch.randperm(num_images_total)[:num_images].tolist()
    return samples_idx


def scan_dataset(dataset_name):
    images_names_list = []

    DL3DV = train_dataset_paths.get('DL3DV', '')
    co3d_data = train_dataset_paths.get('Co3D', '')
    DL3DV_benchmark = test_dataset_paths.get('DL3DV_benchmark', '')
    re10k = train_dataset_paths.get('re10k', '')
    harmony_os_bench0 = test_dataset_paths.get('harmony_os_bench0', '')
    London_data = train_dataset_paths.get('London_data', '')
    Unreal_data = train_dataset_paths.get('Unreal', '')
    brandenburg_data = test_dataset_paths.get('brandenburg_data', '')
    DTU_data = train_dataset_paths.get('DTU_data', '')
    ARKitScenes = train_dataset_paths.get('ARKit', '')
    scannetplusplus = train_dataset_paths.get('scannetplusplus', '')
    synthetic_challenge = test_dataset_paths.get('synthetic_challenge', '')
    wildrgbd = train_dataset_paths.get('wildrgbd', '')
    kitti360 = train_dataset_paths.get('kitti360', '')
    DTU_train = train_dataset_paths.get('DTU_train', '')
    DTU_synthetic_train = train_dataset_paths.get('DTU_synthetic_train', '')
    BlendedMVS = train_dataset_paths.get('BlendedMVS', '')
    harmonyos = train_dataset_paths.get('harmonyos', '')
    
    if dataset_name == 'DL3DV':
        scene_names = []
        corrupted = '3bc16c4c0e9a0e272604a54c2d0b6c5b2193062c4146c5be15f62f542d2f6a6b'
        for k in ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '10K', '11K']:
            base_dir = os.path.join(DL3DV, k)
            if os.path.exists(base_dir):
                scenes = sorted(os.listdir(base_dir))
                scene_names += [os.path.join(base_dir, s, 'images_4') for s in scenes if corrupted not in s]
        #scene_names.remove("/mnt/data/DL3DV-10K/6K/3bc16c4c0e9a0e272604a54c2d0b6c5b2193062c4146c5be15f62f542d2f6a6b/images_4")
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'co3d':
        co3d_objects = ['apple', 'backpack', 'ball', 'banana', 'baseballbat', 'baseballglove', \
        'bench', 'bicycle', 'book', 'bottle', 'bowl', 'broccoli', 'cake', 'car', 'carrot', 'cellphone', \
        'chair', 'couch', 'cup', 'donut', 'frisbee', 'hairdryer', 'handbag', 'hotdog', 'hydrant', 'keyboard', \
        'kite', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'parkingmeter', 'pizza', 'plant', 'remote', \
        'sandwich', 'skateboard', 'stopsign', 'suitcase', 'teddybear', 'toaster', 'toilet', 'toybus', 'toyplane', 'toytrain', \
        'toytruck', 'tv', 'umbrella', 'vase', 'wineglass']
        train_scenes_names = []
        for ob in co3d_objects:
            obj_path = os.path.join(co3d_data, ob)
            if os.path.exists(obj_path):
                scenes_names = sorted(os.listdir(obj_path))
                for junk in ['eval_batches', 'set_lists', 'LICENSE', 'frame_annotations.jgz', 'sequence_annotations.jgz']:
                    if junk in scenes_names: scenes_names.remove(junk)
                train_scenes_names += [os.path.join(obj_path, s, 'images') for s in scenes_names]
        images_names_list = [images_names_from_dir(s) for s in train_scenes_names]
    elif dataset_name == 'DL3DV_benchmark':
        if os.path.exists(DL3DV_benchmark):
            scene_names = sorted(os.listdir(DL3DV_benchmark))
            scene_names = [s for s in scene_names if s != '.cache' and s != 'benchmark-meta.csv']
            scene_names = [os.path.join(DL3DV_benchmark, s, 'nerfstudio/images_4') for s in scene_names]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 're10k':    
        path = os.path.join(re10k, 'train', 'images')
        if os.path.exists(path):
            scenes = sorted(os.listdir(path))
            scene_names = [os.path.join(path, s) for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'HarmonyOS_bench':
        if os.path.exists(harmony_os_bench0):
            scene_names = os.listdir(harmony_os_bench0)
            scene_names = sorted([x for x in scene_names if not x.endswith('zip')])
            scene_names = [os.path.join(harmony_os_bench0, s, 'images') for s in scene_names]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'London':
        if os.path.exists(London_data):
            scene_names = sorted(os.listdir(London_data))
            scene_names = [os.path.join(London_data, s) for s in scene_names]
            images_names_list = [images_names_from_dir(s, filter_str='right') for s in scene_names]
    elif dataset_name == 'Unreal':
        if os.path.exists(Unreal_data):
            base_scenes = sorted(os.listdir(Unreal_data))
            scene_names = [os.path.join(Unreal_data, s, 'Image0') for s in base_scenes] + \
                          [os.path.join(Unreal_data, s, 'Image1') for s in base_scenes]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'brandenburg':
        if os.path.exists(brandenburg_data):
            images_names_list = [images_names_from_dir(brandenburg_data)]
    elif dataset_name == 'DTU':
        if os.path.exists(DTU_data):
            scene_names = sorted(os.listdir(DTU_data))
            for junk in ['README.md', 'SampleSet']:
                if junk in scene_names: scene_names.remove(junk)
            scene_names = [os.path.join(DTU_data, s, 'images') for s in scene_names]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'ARKit':
        scene_names = []
        for split in ['Training', 'Validation']:
            path = os.path.join(ARKitScenes, split)
            if os.path.exists(path):
                scenes = sorted(os.listdir(path))
                scene_names += [os.path.join(path, s, f"{s}_frames", 'lowres_wide') for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "ScanNetplusplus":
        if os.path.exists(scannetplusplus):
            scene_names = sorted(os.listdir(scannetplusplus))
            scene_names = [os.path.join(scannetplusplus, s, 'dslr', 'resized_undistorted_images') for s in scene_names]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'synthetic_challenge':
         if os.path.exists(synthetic_challenge):
            scene_names = sorted(os.listdir(synthetic_challenge))
            scene_names = [os.path.join(synthetic_challenge, s) for s in scene_names]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == 'wildrgbd':
        if os.path.exists(wildrgbd):
            object_names = ['orange','microwave','cup','TV','bottle','banana','donut','potato','pear','cellphone','car','ball', \
                            'flower_pot','pineapple','clock','bucket','cucumber','apple','plane','bus','carrot','peach','box', \
                            'cake','boat','mouse','knife','keyboard','book','detergent','chair','hat','kettle','backpack','bowl',
                            'handbag']
            scene_names = []
            for o in object_names:
                object_scene_names = sorted(os.listdir(os.path.join(wildrgbd, o, 'scenes')))
                object_scene_names = [os.path.join(wildrgbd, o, 'scenes', s, 'rgb') for s in object_scene_names]
                scene_names += object_scene_names

            images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "kitti360":
        scenes = sorted(os.listdir(kitti360))
        scenes.remove('dataset_metadata.json')
        scene_names = [os.path.join(kitti360, s) for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "DTU_train":
        scenes = sorted(os.listdir(DTU_train))
        scenes.remove('dataset_metadata.json')
        scene_names = [os.path.join(DTU_train, s) for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "DTU_synthetic_train":
        scenes = sorted(os.listdir(DTU_synthetic_train))
        scenes.remove('dataset_metadata.json')
        scene_names = [os.path.join(DTU_synthetic_train, s) for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "BlendedMVS":
        scenes = sorted(os.listdir(BlendedMVS))
        scenes.remove('dataset_metadata.json')
        scene_names = [os.path.join(BlendedMVS, s) for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "harmonyos":
        scenes = sorted(os.listdir(harmonyos))
        scenes.remove('dataset_metadata.json')
        scene_names = [os.path.join(harmonyos, s) for s in scenes]
        images_names_list = [images_names_from_dir(s) for s in scene_names]
    elif dataset_name == "minimal_dataset":
        minimal_dataset_path = train_dataset_paths.get('minimal_dataset', '')
        if os.path.exists(minimal_dataset_path):
            scenes = sorted(os.listdir(minimal_dataset_path))
            scene_names = [os.path.join(minimal_dataset_path, s) for s in scenes if os.path.isdir(os.path.join(minimal_dataset_path, s))]
            images_names_list = [images_names_from_dir(s) for s in scene_names]
    
    else:
        if os.path.exists(dataset_name):
            images_names_list = [images_names_from_dir(dataset_name)]

    return [s for s in images_names_list if len(s) >= 2]

class InfiniteLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch

# --- Helper Function for Parallel Initialization ---
def init_single_dataset(name, step_range, sampling_priority,shared_args):
    scan_wrapper = lambda n=name: scan_dataset(n) # Ensure scan_dataset is imported
    ds = TrainingDataset(
        name=name,
        scan_fn=scan_wrapper,
        image_num_range=shared_args['image_num_range'], # Updated key
        num_target=shared_args['num_target'],
        target_mode=shared_args['target_mode'],
        sampling_mode=shared_args['sampling_mode'],
        loading_mode=shared_args['loading_mode'],
        load_size=shared_args['load_size'],
        step_range=step_range,
        sampling_priority=sampling_priority,
        cache_dir='./dataset_cache/'
    )
    return ds

# --- Main Factory Function ---
def create_train_dataloader(dataset_names,
                            num_target=0,
                            image_num_range=(1, 24),
                            target_mode="interpolation",
                            sampling_mode='linear_close',
                            loading_mode='vggt_crop',
                            load_size=518,
                            max_pixels_per_gpu=24*518*518,
                            num_gpus=torch.cuda.device_count(),
                            num_workers=8):
    
    # Hardcoded configuration for step ranges per dataset
    all_dataset_configs = [
        {'name': 'DL3DV', 'step_range': (5, 10), 'sampling_priority': 7.},
        {'name': 'co3d', 'step_range': (5, 10), 'sampling_priority': 4.},
        {'name': 'London', 'step_range': (10, 20), 'sampling_priority': 1.},
        {'name': 'ARKit', 'step_range': (5, 10), 'sampling_priority': 4.},
        {'name': 'ScanNetplusplus', 'step_range': (2, 8), 'sampling_priority': 3.},
        {'name': 'Unreal', 'step_range': (2, 5), 'sampling_priority': 1.},
        {'name': 're10k', 'step_range': (10, 30), 'sampling_priority': 10.},
        {'name': 'wildrgbd', 'step_range': (5, 10), 'sampling_priority': 5.},
        {'name': 'kitti360', 'step_range': (10, 30), 'sampling_priority': 1.},
        {'name': 'DTU_train', 'step_range': (1, 3), 'sampling_priority': 2.},
        {'name': 'DTU_synthetic_train', 'step_range': (1, 3), 'sampling_priority': 2.},
        {'name': 'BlendedMVS', 'step_range': (2, 8), 'sampling_priority': 2.},
        {'name': 'harmonyos', 'step_range': (5, 10), 'sampling_priority': 1.},
        {'name': 'minimal_dataset', 'step_range': (1, 6), 'sampling_priority': 1.},
    ]
    "kitti360""DTU_train""DTU_synthetic_train""BlendedMVS""harmonyos"
    # 1. Filter configs based on the requested 'dataset_names' list
    relevant_configs = [cfg for cfg in all_dataset_configs if cfg['name'] in dataset_names]
    
    # 2. Bundle shared arguments to pass to the helper
    shared_args = {
        'image_num_range': image_num_range,
        'num_target': num_target,
        'target_mode': target_mode,
        'sampling_mode': sampling_mode,
        'loading_mode': loading_mode,
        'load_size': load_size
    }

    datasets = []
    
    # 3. Initialize Datasets in Parallel
    # We use ThreadPoolExecutor because the bottleneck is I/O (disk scanning)
    print(f"Initializing {len(relevant_configs)} datasets in parallel...")
    
    # Safety: Ensure we don't spawn more threads than needed
    max_threads = min(8, len(relevant_configs)) if len(relevant_configs) > 0 else 1
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit all initialization tasks
        future_to_name = {
            executor.submit(init_single_dataset, cfg['name'], cfg['step_range'], cfg['sampling_priority'], shared_args): cfg['name']
            for cfg in relevant_configs
        }
        
        # Collect results as they finish
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                ds = future.result()
                datasets.append(ds)
            except Exception as exc:
                print(f"[{name}] Generated an exception during initialization: {exc}")
                # You might want to raise here depending on how strict you want to be
                # raise exc 

    # 4. Sort datasets deterministically
    # Parallel execution finishes in random order. Sorting by name ensures
    # the MultiTrainingDataset is built identically across runs (important for resuming).
    datasets.sort(key=lambda d: d.name)

    multi_dataset = MultiTrainingDataset(datasets)
    
    sampler = DynamicBatchSampler(
        multi_dataset, 
        max_pixels_per_gpu=max_pixels_per_gpu, 
        image_num_range=image_num_range, 
        num_gpus=num_gpus
    )
    
    loader = DataLoader(
        multi_dataset, 
        batch_sampler=sampler, 
        num_workers=num_workers
    )
    
    return loader