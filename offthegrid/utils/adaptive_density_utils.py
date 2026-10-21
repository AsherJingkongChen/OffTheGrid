import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange

def get_adaptive_patches_from_images(images, mode="adaptive_entropy_image", conf=None, num_scales=3, percentages=[0.55, 0.30, 0.15], patch_size=14):
    """
    images: tensor B N C H W
    mode: str 
        Fixed: 'low', 'medium', 'high', 'random'
        Adaptive: 'adaptive_entropy_image', 'adaptive_variance_image', 'adaptive_conf_image', 'adaptive_entropy_conf_image'
                  (and their '_scene' counterparts)
    conf: None or tensor (B*N) 1 H W (ranging 0 to 1)
    num_scales: int
    percentages: list of len num_scales, must sum to 1
    patch_size: int

    return:
    masks dict of tensors B D (or B*N D)
    num_patchs list [D1 D2 ... D_num_scales]
    """
    B, N, C, H, W = images.shape
    device = images.device
    
    assert len(percentages) == num_scales, "Length of percentages must match num_scales"
    
    # Calculate base number of patches
    patches_per_frame = (H // patch_size) * (W // patch_size)
    total_patches = patches_per_frame * N if 'scene' in mode else patches_per_frame
    
    # --- 1. Handle Fixed Modes ---
    if mode in ['low', 'medium', 'high']:
        num_patchs = [0] * num_scales
        if mode == 'low':
            target_idx = 0
        elif mode == 'high':
            target_idx = num_scales - 1
        else: # medium
            target_idx = num_scales // 2
            
        num_patchs[target_idx] = total_patches
        
        masks = {str(i): [] for i in range(num_scales)}
        batch_dim = B if 'scene' in mode else B * N
        masks[str(target_idx)] = torch.arange(total_patches)[None].expand(batch_dim, -1).to(device)
        return masks, num_patchs

    # --- 2. Calculate Scores for Sorting Modes ---
    if 'adaptive' in mode or mode == 'random':
        if mode == 'random':
            # Generate random scores to force a random distribution
            batch_dim = B if 'scene' in mode else B * N
            scores = torch.rand((batch_dim, total_patches), device=device)
            
        else:
            # Prepare images: (B*N) C H W
            images_flat = rearrange(images.clone().detach() * 255, 'B N C H W -> (B N) C H W')
            
            scores = None
            
            # A. Compute Entropy
            if 'entropy' in mode:
                entropy_maps = compute_patch_metrics_batched(images_flat, patch_size, metric='entropy')[patch_size]
                scores = entropy_maps
                
            # B. Compute Variance
            if 'variance' in mode:
                variance_maps = compute_patch_metrics_batched(images_flat, patch_size, metric='variance')[patch_size]
                scores = variance_maps
                
            # C. Compute Confidence
            if 'conf' in mode:
                assert conf is not None, "conf tensor must be provided for conf modes"
                # Average pooling to get mean confidence per patch
                conf_pooled = F.avg_pool2d(conf, kernel_size=patch_size, stride=patch_size).squeeze(1)
                
                if scores is not None:
                    scores = scores * conf_pooled # adaptive_entropy_conf_image
                else:
                    scores = conf_pooled          # adaptive_conf_image

            # Format scores based on image vs scene routing
            if 'image' in mode: 
                scores = rearrange(scores, '(B N) H W -> (B N) (H W)', B=B, N=N)
            else: # scene
                scores = rearrange(scores, '(B N) H W -> B (N H W)', B=B, N=N)

        # --- 3. Sort and Distribute ---
        # Note: pad_value is set to -1 in metric computation so padded regions naturally fall to the end
        _, indices = torch.sort(scores, dim=1, descending=False)
        
        # Calculate dynamic bucket capacities
        num_patchs = [int(total_patches * p) for p in percentages]
        num_patchs[-1] = total_patches - sum(num_patchs[:-1]) # Fix rounding errors
        
        masks = {}
        start_idx = 0
        # For adaptive, bucket 0 is usually lowest resolution, so we assign highest scores to the LAST bucket.
        # However, to keep it generalized, we will map them sequentially. You can reverse the list if you 
        # want bucket 0 to be the high-compute bucket.
        for i, count in enumerate(num_patchs):
            masks[str(i)] = indices[:, start_idx : start_idx + count]
            start_idx += count
            
        return masks, num_patchs

def compute_patch_metrics_batched(images, patch_size=16, num_scales=1, bins=256, metric='entropy'):
    """
    Computes metrics (entropy or variance) for multiple patch sizes in a batch of images.
    Memory-optimized to avoid massive VRAM allocations.
    """
    batch_size, channels, H, W = images.shape
    device = images.device
    
    # Fast Grayscale Conversion
    if channels == 3:
        grayscale_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
        grayscale_images = (images * grayscale_weights).sum(dim=1, keepdim=True)
    else:
        grayscale_images = images[:, 0:1]
        
    batch_metric_maps = {}
    patch_sizes = [patch_size * (2**i) for i in range(num_scales)]
    
    for ps in patch_sizes:
        num_patches_h = (H + ps - 1) // ps
        num_patches_w = (W + ps - 1) // ps
        
        pad_h = num_patches_h * ps - H
        pad_w = num_patches_w * ps - W
        padded_images = F.pad(grayscale_images, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        if metric == 'variance':
            # Mathematical variance: Var(X) = E[X^2] - (E[X])^2
            # Hardware accelerated via avg_pool2d
            mu = F.avg_pool2d(padded_images, kernel_size=ps, stride=ps)
            mu_sq = F.avg_pool2d(padded_images**2, kernel_size=ps, stride=ps)
            metric_map = (mu_sq - mu**2).squeeze(1)
            
        elif metric == 'entropy':
            # 1. Unfold into patches
            patches = padded_images.squeeze(1).unfold(1, ps, ps).unfold(2, ps, ps)
            flat_patches = patches.reshape(batch_size, num_patches_h, num_patches_w, ps*ps)
            
            # 2. Quantize
            flat_patches_int = (flat_patches * (bins / 256.0)).long().clamp(0, bins-1)
            reshaped_patches = flat_patches_int.reshape(-1, ps*ps) # Shape: [B * H_p * W_p, ps*ps]
            
            # 3. VRAM-Safe Histogram via scatter_add_
            num_total_patches = reshaped_patches.size(0)
            histograms = torch.zeros(num_total_patches, bins, device=device)
            ones = torch.ones_like(reshaped_patches, dtype=torch.float)
            histograms.scatter_add_(1, reshaped_patches, ones)
            
            # 4. Probabilities and Shannon Entropy
            histograms = histograms.reshape(batch_size, num_patches_h, num_patches_w, bins)
            probabilities = histograms / (ps * ps)
            
            epsilon = 1e-10
            metric_map = -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=3)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # IMPORTANT: Pad value set to -1 so padded patches are sorted to the bottom
        if pad_h > 0:
            metric_map[:, -1, :] = -1 
        if pad_w > 0:
            metric_map[:, :, -1] = -1
            
        batch_metric_maps[ps] = metric_map
        
    return batch_metric_maps
    
def visualize_adaptive_patches(images, masks, patch_size=14, alpha=0.5):
    """
    Visualizes the patches by overlaying colors on the original images.
    Dynamically scales to any number of scales.
    Default 3-scale colors: Blue (0), Yellow (1), Red (2)
    """
    B, N, C, H, W = images.shape
    device = images.device

    patches_h = H // patch_size
    patches_w = W // patch_size
    patches_per_img = patches_h * patches_w

    color_palette = [
        [0.0, 0.0, 1.0], # Blue 
        [1.0, 1.0, 0.0], # Yellow
        [1.0, 0.0, 0.0], # Red 
        [0.0, 1.0, 0.0], # Green
        [1.0, 0.0, 1.0], # Magenta
        [0.0, 1.0, 1.0], # Cyan
    ]

    # Initialize correctly as 4D to keep memory boundaries safe
    # Shape: (B, N, 3, patches_per_img)
    canvas = torch.zeros((B, N, 3, patches_per_img), device=device)
    alpha_mask = torch.zeros((B, N, 1, patches_per_img), device=device)

    for key_idx, (key, mask) in enumerate(masks.items()):
        if mask is None or len(mask) == 0 or mask.numel() == 0:
            continue
        
        color = torch.tensor(color_palette[int(key) % len(color_palette)], device=device).view(1, 3, 1)

        if mask.shape[0] == B:
            # --- Scene Mode ---
            # Rearrange safely permutes axes so channels are contiguous
            canvas_scene = rearrange(canvas, 'B N C P -> B C (N P)')
            alpha_scene = rearrange(alpha_mask, 'B N C P -> B C (N P)')
            
            mask_expanded = mask.unsqueeze(1).expand(-1, 3, -1)
            mask_alpha_expanded = mask.unsqueeze(1).expand(-1, 1, -1)
            
            canvas_scene.scatter_(2, mask_expanded, color.expand(B, 3, mask.shape[1]))
            alpha_scene.scatter_(2, mask_alpha_expanded, torch.ones_like(mask_alpha_expanded, dtype=torch.float))
            
            # Put safely back into B N C P
            canvas = rearrange(canvas_scene, 'B C (N P) -> B N C P', N=N)
            alpha_mask = rearrange(alpha_scene, 'B C (N P) -> B N C P', N=N)
            
        else:
            # --- Image Mode ---
            canvas_img = rearrange(canvas, 'B N C P -> (B N) C P')
            alpha_img = rearrange(alpha_mask, 'B N C P -> (B N) C P')
            
            mask_expanded = mask.unsqueeze(1).expand(-1, 3, -1)
            mask_alpha_expanded = mask.unsqueeze(1).expand(-1, 1, -1)
            
            canvas_img.scatter_(2, mask_expanded, color.expand(B * N, 3, mask.shape[1]))
            alpha_img.scatter_(2, mask_alpha_expanded, torch.ones_like(mask_alpha_expanded, dtype=torch.float))
            
            canvas = rearrange(canvas_img, '(B N) C P -> B N C P', B=B, N=N)
            alpha_mask = rearrange(alpha_img, '(B N) C P -> B N C P', B=B, N=N)

    # Reshape the sequence back into a 2D spatial grid for upsampling
    canvas_2d = rearrange(canvas, 'B N C (H W) -> (B N) C H W', H=patches_h, W=patches_w)
    alpha_2d = rearrange(alpha_mask, 'B N C (H W) -> (B N) C H W', H=patches_h, W=patches_w)

    # Upsample the 1x1 patch pixels to full patch_size rectangles
    overlays = F.interpolate(canvas_2d, scale_factor=patch_size, mode='nearest')
    mask_exists = F.interpolate(alpha_2d, scale_factor=patch_size, mode='nearest')

    images_flat = rearrange(images.clone().detach(), 'B N C H W -> (B N) C H W')
    
    result_flat = torch.where(mask_exists > 0.5, images_flat * (1 - alpha) + overlays * alpha, images_flat)

    result = rearrange(result_flat, '(B N) C H W -> B N C H W', B=B, N=N)
    
    return result

def create_synthetic_batch(B=1, N=2, C=3, H=224, W=224, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Creates a synthetic tensor with distinct spatial frequencies to test adaptive routing.
    Image 0: Left half flat (low entropy), Right half noise (high entropy).
    Image 1: Top half smooth gradient (low variance), Bottom half checkerboard (high variance).
    """
    images = torch.zeros((B, N, C, H, W), device=device)
    
    for b in range(B):
        # Frame 0: Flat vs Noise
        images[b, 0, :, :, :W//2] = 0.5  # Flat gray
        images[b, 0, :, :, W//2:] = torch.rand((C, H, W//2), device=device) # High frequency noise
        
        # Frame 1: Gradient vs Checkerboard
        if N > 1:
            # Gradient
            gradient = torch.linspace(0, 1, H//2, device=device).view(1, H//2, 1).expand(C, H//2, W)
            images[b, 1, :, :H//2, :] = gradient
            
            # Checkerboard (High variance)
            y, x = torch.meshgrid(torch.arange(H//2, device=device), torch.arange(W, device=device), indexing='ij')
            checker = ((x // 16) + (y // 16)) % 2
            images[b, 1, :, H//2:, :] = checker.unsqueeze(0).expand(C, H//2, W).float()
            
    return images

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # 1. Setup parameters
    B, N, C, H, W = 1, 2, 3, 224, 224
    patch_size = 16
    num_scales = 3
    percentages = [0.5, 0.3, 0.2] # 50% Low (Blue), 30% Medium (Yellow), 20% High (Red)
    
    # Generate synthetic data with clear textures
    images = create_synthetic_batch(B, N, C, H, W, device)
    
    # Modes to test (excluding confidence-based modes since we don't have a conf map)
    modes = [
        'low', 'medium', 'high', 'random',
        'adaptive_entropy_image', 'adaptive_variance_image',
        'adaptive_entropy_scene', 'adaptive_variance_scene'
    ]
    
    # Set up matplotlib figure
    fig, axes = plt.subplots(len(modes), N, figsize=(5 * N, 4 * len(modes)))
    if N == 1:
        axes = axes[:, None] # Ensure 2D array for indexing if N=1
        
    fig.suptitle(f"Adaptive Patch Routing Visualization\nRed=High Compute ({percentages[2]*100}%), Yellow=Med ({percentages[1]*100}%), Blue=Low ({percentages[0]*100}%)", fontsize=16)
    
    # 2. Process and visualize each mode
    for i, mode in enumerate(modes):
        print(f"Processing mode: {mode}...")
        
        # Get masks
        masks, num_patches = get_adaptive_patches_from_images(
            images, 
            mode=mode, 
            num_scales=num_scales, 
            percentages=percentages, 
            patch_size=patch_size
        )
        
        # Visualize
        vis_images = visualize_adaptive_patches(images, masks, patch_size=patch_size, alpha=0.6)
        
        # 3. Plotting
        # Move back to CPU and reshape for matplotlib (B N C H W -> B N H W C)
        vis_cpu = rearrange(vis_images.cpu().clamp(0, 1), 'B N C H W -> B N H W C')
        
        for n in range(N):
            ax = axes[i, n]
            # Just plotting the first batch item for simplicity
            ax.imshow(vis_cpu[0, n].numpy())
            ax.axis('off')
            if n == 0:
                ax.set_title(f"Mode: {mode}\nFrame {n}", loc='left')
            else:
                ax.set_title(f"Frame {n}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    save_path = "routing_visualization.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nSuccess! Visualization saved to {save_path}")
    
    # If running in an interactive window (Jupyter/IDE), display it
    plt.show()

if __name__ == "__main__":
    main()