import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import cv2

def get_target_dims(w, h, mode, target_size, patch_size):
    """Calculates new dimensions based on the selected strategy."""
    if mode == 'resize':
        # Direct resize to square, ignoring aspect ratio
        return target_size, target_size
    
    elif mode == 'no_crop':
        # Resize longest dimension to target_size, maintain aspect ratio
        if w >= h:
            new_w = target_size
            new_h = round(h * (target_size / w) / patch_size) * patch_size
        else:
            new_h = target_size
            new_w = round(w * (target_size / h) / patch_size) * patch_size
        return new_w, new_h

    elif mode in ['vggt_crop', 'vggt_pad']:
        if mode == 'vggt_crop':
            # Primary constraint is width = target_size
            new_w = target_size
            new_h = round(h * (target_size / w) / patch_size) * patch_size
        else: # vggt_pad
            # Ensure the largest side fits target_size
            scale = target_size / max(w, h)
            new_w = round(w * scale / patch_size) * patch_size
            new_h = round(h * scale / patch_size) * patch_size
        return new_w, new_h

    elif mode == 'center_crop':
        # Resize shortest side to target_size
        scale = target_size / min(w, h)
        return round(w * scale), round(h * scale)

    elif mode == 'closest':
        # Snap to nearest patch_size multiple
        new_w = max(round(w / patch_size) * patch_size, patch_size)
        new_h = max(round(h / patch_size) * patch_size, patch_size)
        return new_w, new_h

    raise ValueError(f"Unknown mode: {mode}")

def load_and_preprocess_images(path_list, mode='closest', target_size=518, patch_size=None):
    if not path_list:
        raise ValueError("Path list is empty")

    if patch_size is None:
        if target_size % 14 == 0:
            patch_size = 14
        else:
            patch_size = 16

    processed_tensors = []
    # original_shapes = [] # Uncomment if you need to return this
    shapes_after_resize = set()
    to_tensor = TF.ToTensor()

    for path in path_list:
        if path.lower().endswith('.npy') or 'depth_gt' in path:
            # --- Load Depth Map ---
            if path.lower().endswith('.npy'):
                depth_map = np.load(path).astype(np.float32)
                h, w = depth_map.shape[:2]
            else:
                depth_map = cv2.imread(path,-1).astype(np.float32) / 1000
                h, w = depth_map.shape
            
            # original_shapes.append([h, w])
            
            # Ensure (C, H, W) -> (1, 1, H, W) for interpolate
            data_tensor = torch.from_numpy(depth_map)
            if data_tensor.ndim == 2:
                data_tensor = data_tensor.unsqueeze(0)
            data_tensor = data_tensor.unsqueeze(0)
            is_depth = True
        else:
            # --- Load Image ---
            img = Image.open(path)
            w, h = img.size
            # original_shapes.append([h, w])
            
            if img.mode == "RGBA":
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = img.convert("RGB")
            data_tensor = to_tensor(img).unsqueeze(0) # (1, 3, H, W)
            is_depth = False

        if max(w,h) != target_size or w % patch_size != 0 or h %patch_size != 0: 
            # 1. Calculate and Execute Resize
            new_w, new_h = get_target_dims(w, h, mode, target_size, patch_size)
            data_tensor = F.interpolate(data_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            data_tensor = data_tensor.squeeze(0) # (C, H, W)

            # 2. Strategy Post-Processing (Crops/Pads)
            if mode == 'vggt_crop' and data_tensor.shape[1] > target_size:
                start_y = (data_tensor.shape[1] - target_size) // 2
                data_tensor = data_tensor[:, start_y : start_y + target_size, :]
                
            elif mode == 'center_crop':
                th, tw = data_tensor.shape[1], data_tensor.shape[2]
                start_y, start_x = (th - target_size) // 2, (tw - target_size) // 2
                data_tensor = data_tensor[:, start_y:start_y+target_size, start_x:start_x+target_size]
                
            elif mode == 'vggt_pad':
                h_pad, w_pad = target_size - data_tensor.shape[1], target_size - data_tensor.shape[2]
                fill = 0.0 if is_depth else 1.0
                data_tensor = F.pad(data_tensor, (w_pad//2, w_pad - w_pad//2, h_pad//2, h_pad - h_pad//2), value=fill)
        else:
            data_tensor = data_tensor.squeeze(0)
        # Note: 'no_crop' requires no specific post-processing here.
        # It relies on get_target_dims to set the correct size, and the 
        # final batch normalization below to pad if mixed aspect ratios exist.

        processed_tensors.append(data_tensor)
        shapes_after_resize.add((data_tensor.shape[1], data_tensor.shape[2]))

    # 3. Final Batch Normalization (Padding if shapes differ in the same batch)
    if len(shapes_after_resize) > 1:
        max_h = max(s[0] for s in shapes_after_resize)
        max_w = max(s[1] for s in shapes_after_resize)
        batch_tensors = []
        for t in processed_tensors:
            h_p, w_p = max_h - t.shape[1], max_w - t.shape[2]
            if h_p > 0 or w_p > 0:
                # Default to 0 padding for batch consistency
                t = F.pad(t, (w_p//2, w_p - w_p//2, h_p//2, h_p - h_p//2), value=0.0)
            batch_tensors.append(t)
        processed_tensors = batch_tensors

    return torch.stack(processed_tensors)


def adapt_intrinsics(source_intr, source_mode, target_mode, 
                     orig_h, orig_w, 
                     source_size=518, target_size=518, 
                     patch_size=14):
    """
    Adjusts the predicted source intrinsics to match a target loading mode.
    """
    # 1. Get the final pixel dimensions for both modes
    src_h, src_w = get_target_dims(orig_h, orig_w, source_mode, source_size, patch_size)
    tgt_h, tgt_w = get_target_dims(orig_h, orig_w, target_mode, target_size, patch_size)

    # 2. Calculate scaling ratios between target and source pixels
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h

    target_intr = source_intr.clone()
    
    # 3. Update Focal Lengths
    target_intr[..., 0, 0] *= scale_x
    target_intr[..., 1, 1] *= scale_y
    
    # 4. Set Principal Point to exactly the middle of the target image
    target_intr[..., 0, 2] = tgt_w / 2.0
    target_intr[..., 1, 2] = tgt_h / 2.0
    
    return target_intr

def adapt_depth_maps(source_depth, source_mode, target_mode, 
                     orig_h, orig_w, 
                     source_size=518, target_size=518, 
                     patch_size=14):
    """
    Transforms depth maps from one loading strategy to another.
    """
    if source_depth.ndim == 3:
        source_depth = source_depth.unsqueeze(0)
        
    def get_resize_only_dims(h, w, mode, size, patch_size):
        if mode == 'resize': return size, size
        
        if mode == 'no_crop':
            # Logic is same as final dims for no_crop (no post-resize cropping)
            if w >= h:
                new_w = size
                new_h = round(h * (size / w) / patch_size) * patch_size
            else:
                new_h = size
                new_w = round(w * (size / h) / patch_size) * patch_size
            return new_h, new_w

        if mode == 'vggt_crop' or mode == 'vggt_pad':
            if mode == 'vggt_crop':
                new_w = size
                new_h = round(h * (size / w) / patch_size) * patch_size
            else: # vggt_pad
                scale = size / max(h, w)
                new_w = round(w * scale / patch_size) * patch_size
                new_h = round(h * scale / patch_size) * patch_size
            return new_h, new_w
        if mode == 'center_crop':
            scale = size / min(h, w)
            return round(h * scale), round(w * scale)
        if mode == 'closest':
            return max(round(h / patch_size) * patch_size, patch_size), \
                   max(round(w / patch_size) * patch_size, patch_size)
        return h, w

    # Dimensions actually present in the source_depth tensor
    src_final_h, src_final_w = get_target_dims(orig_h, orig_w, source_mode, source_size, patch_size)
    # Dimensions if we hadn't cropped/padded in source
    src_resize_h, src_resize_w = get_resize_only_dims(orig_h, orig_w, source_mode, source_size, patch_size)
    
    # Dimensions for target
    tgt_final_h, tgt_final_w = get_target_dims(orig_h, orig_w, target_mode, target_size, patch_size)
    tgt_resize_h, tgt_resize_w = get_resize_only_dims(orig_h, orig_w, target_mode, target_size, patch_size)

    # 2. Reverse Source Post-processing (Un-crop / Un-pad)
    h_diff = src_resize_h - src_final_h
    w_diff = src_resize_w - src_final_w
    working_depth = F.pad(source_depth, (w_diff//2, w_diff - w_diff//2, h_diff//2, h_diff - h_diff//2), value=0)

    # 3. Scale to Target Resize Dimensions
    working_depth = F.interpolate(working_depth, size=(tgt_resize_h, tgt_resize_w), mode='bilinear', align_corners=False)

    # 4. Apply Target Post-processing (Crop / Pad)
    if tgt_final_h <= tgt_resize_h and tgt_final_w <= tgt_resize_w:
        # Target needs a crop
        sy = (tgt_resize_h - tgt_final_h) // 2
        sx = (tgt_resize_w - tgt_final_w) // 2
        final_depth = working_depth[:, :, sy:sy+tgt_final_h, sx:sx+tgt_final_w]
    else:
        # Target needs padding
        ph = tgt_final_h - tgt_resize_h
        pw = tgt_final_w - tgt_resize_w
        final_depth = F.pad(working_depth, (pw//2, pw - pw//2, ph//2, ph - ph//2), value=0)

    return final_depth

def indices_to_select(num_total_frames,num_subsampled_images):
    if num_subsampled_images < num_total_frames:
        bin_size = num_total_frames // num_subsampled_images
        selected = [k for k in range(num_total_frames) if k%bin_size==0]
        if len(selected) > num_subsampled_images:
            num_to_remove = len(selected) - num_subsampled_images
            to_remove = [k*2 + 1 for k in range(num_to_remove)]
            selected = [selected[k] for k in range(len(selected)) if k not in to_remove]
    else:
        selected = list(range(num_total_frames))
    return selected



if __name__ == "__main__":
    test_adapt_intrinsics()
