import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
from einops import rearrange
from plyfile import PlyData, PlyElement
from src.utils.geometry_utils import depth_to_cam_coords_points_torch, closed_form_inverse_se3, pred_to_cam_coords, unproject_depth_map_to_point_map_torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import tempfile
import subprocess
import os
import cv2
import json

from gsplat.rendering import rasterization

def depth_to_colormap(depth_map: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Converts a 1-channel depth map (H, W) or (1, H, W) to a 3-channel colormap tensor (3, H, W).

    Args:
        depth_map (torch.Tensor): A PyTorch tensor representing the depth map.
                                  Expected shape is (H, W) or (1, H, W).
        normalize (bool): If True, normalizes the depth map to [0, 1] before colormapping.
                          This is generally recommended for consistent visualization.

    Returns:
        torch.Tensor: A 3-channel PyTorch tensor representing the colormapped image (3, H, W).
                      The values will be in the range [0, 1].
    """
    if depth_map.dim() == 3 and depth_map.shape[0] == 1:
        depth_map = depth_map.squeeze(0) # Remove the channel dimension if present

    if depth_map.dim() != 2:
        raise ValueError(f"Expected depth_map to be 2D (H, W) or 3D (1, H, W), but got shape {depth_map.shape}")

    # Convert to numpy for matplotlib colormapping
    depth_np = depth_map.detach().cpu().numpy()

    if normalize:
        # Normalize depth values to [0, 1] for consistent colormapping
        min_val = np.min(depth_np)
        max_val = np.max(depth_np)
        if max_val > min_val:
            depth_np = (depth_np - min_val) / (max_val - min_val)
        else:
            # Handle the case where all depth values are the same
            depth_np = np.zeros_like(depth_np)

    # Apply a colormap (e.g., 'viridis', 'jet', 'plasma', 'inferno')
    # 'viridis' is often a good choice as it's perceptually uniform.
    colormap = plt.cm.viridis(depth_np)

    # colormap will be (H, W, 4) with RGBA. We only need RGB.
    colormap_rgb = colormap[..., :3]

    # Convert numpy array back to torch tensor and reorder dimensions to (C, H, W)
    colormap_tensor = torch.from_numpy(colormap_rgb).permute(2, 0, 1).float()

    return colormap_tensor



def render_nvs_masks(depths, extrinsics, intrinsics, add_safety_gaussian=False):

    B, V, C, H, W = depths.shape
    nvs_masks = torch.zeros_like(depths)
    
    geometry_gaussians = create_geometry_gaussians(depths, extrinsics, intrinsics)
    sources_gaussians = {}
    valid_scenes = [p for p in range(B)]
    for b in range(B):
        for v in range(V):
            try:
                source_views = [a for a in range(V) if a != v]
                for att in geometry_gaussians:
                    sources_gaussians[att] = geometry_gaussians[att][b,source_views]
                    if 'features' in att:
                        sources_gaussians[att] = rearrange(sources_gaussians[att], 'V N S D -> (V N) S D').float()
                    else:
                        sources_gaussians[att] = rearrange(sources_gaussians[att], 'V N D -> (V N) D').float()
                
                if add_safety_gaussian: #add 1 safety gaussian from target view
                    for att in sources_gaussians:
                        if att != 'depths':
                            idx = geometry_gaussians['opacities'].shape[2]//2
                            if att in ['opacities','conf']:
                                sources_gaussians[att] = torch.cat([sources_gaussians[att],torch.ones_like(geometry_gaussians[att][b,v,idx:idx+1])],dim=0)
                            else:
                                sources_gaussians[att] = torch.cat([sources_gaussians[att],geometry_gaussians[att][b,v,idx:idx+1]],dim=0)
                
                rendered_image, blend_alphas, blend_semantics, blend_depths, blend_normals, extra_data = render(sources_gaussians,extrinsics[b,v],intrinsics[b,v],H,W,conf_fusion=False,render_depth_normal=False,confidence_visualization=False)
                if blend_alphas is not None:
                    nvs_masks[b,v] = blend_alphas    
                else:
                    print('rendering fail, no overlap, skip scene')
                    if b in valid_scenes:
                        valid_scenes.remove(b)
                    #nvs_masks[b,v,0,0,0] = 1
            except:
                print('rendering fail, no overlap, skip scene')
                if b in valid_scenes:
                    valid_scenes.remove(b)
    return nvs_masks, valid_scenes


def create_geometry_gaussians(depths, extrinsics, intrinsics):

    B, V, C, H, W = depths.shape
    attribute_dict = {}

    means = torch.zeros(B, V, H*W, 3).float().to(depths.device)
    for b in range(B):
        points = unproject_depth_map_to_point_map_torch(depths[b,:,0], extrinsics[b], intrinsics[b])
        means[b] = rearrange(points, 'V H W D -> V (H W) D')
    attribute_dict["means"] = means#unproject_depth_map_to_point_map_torch(depths, extrinsics, intrinsics)
    attribute_dict['quats'] = torch.zeros(B, V, H*W, 4).float().to(depths.device)
    attribute_dict['quats'][...,:1] = 1
    attribute_dict['scales'] = torch.log(torch.ones(B, V, H*W, 3).float().to(depths.device) * 0.003)
    attribute_dict['opacities'] = torch.ones(B, V, H*W, 1).float().to(depths.device) * 1000
    attribute_dict['features_sh'] = torch.zeros(B,V,H*W,0,3).float().to(depths.device)
    attribute_dict['features_dc'] = RGB2SH(torch.zeros(B,V,H*W,1,3).float().to(depths.device))
    return attribute_dict

def render(gaussians,
           extrinsics,
           intrinsics,
           image_height,
           image_width,
           bg_color: Tensor = None,
           return_abs_grad=False,
           render_depth_normal=True,
           conf_fusion=False,
           confidence_visualization=False,
           rasterizer='gsplat',
           device='cuda'
           ):
    
    active_sh_degree = 0
    
    if conf_fusion:
        opacities = torch.sigmoid(gaussians['opacities']) * gaussians['conf']
        opacities = inverse_sigmoid(opacities,epsilon=1e-5)
    else:
        opacities = gaussians['opacities']

    if confidence_visualization:
        #opacities = inverse_sigmoid(torch.ones_like(gaussians['opacities']),epsilon=1e-5)
        conf_color = torch.tensor([5/255,171/255,28/255]).to(gaussians['conf'].device)
        sh_colors = RGB2SH(conf_color[None]*gaussians['conf'][...,None])
    else:
        sh_colors = gaussians['features_dc']

    if rasterizer == 'gs_render':
        raise NotImplementedError

    elif rasterizer == 'gsplat':
        packed = False
        render_mode = "RGB+D" if render_depth_normal else "RGB"
        rasterize_mode="classic"#"antialiased"#
        padded_extr = torch.cat([extrinsics,torch.tensor([[0.,0.,0.,1]]).to(extrinsics.device)], dim=0)[None]
        opacities = torch.sigmoid(opacities.float())[:,0]
        rast_colors, rast_alphas, info = rasterization(
            means=gaussians['means'],
            quats=gaussians['quats'],
            scales=torch.exp(gaussians['scales']),# + 0.0005,
            opacities=opacities,
            colors=SH2RGB(sh_colors)[:,0],
            viewmats=padded_extr,  # [C, 4, 4]
            Ks=intrinsics[None],  # [C, 3, 3]
            width=image_width,
            height=image_height,
            packed=packed,
            absgrad=return_abs_grad,
            backgrounds=bg_color[None],
            render_mode=render_mode,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            with_eval3d=render_depth_normal,
            return_normals=render_depth_normal,
            with_ut=False,
            global_z_order=True,
            camera_model="pinhole")
            #eps2d=0.0)
        
        if render_mode == "RGB+D":
            blend_depths = rast_colors[0,...,-1:].permute(2,0,1)
            blend_colors = rast_colors[0,...,:-1].permute(2,0,1)
            blend_normals = info["normals"][0].permute(2,0,1)
            blend_normals = torch.nn.functional.normalize(blend_normals, dim=0)
        else:
            blend_colors = rast_colors[0].permute(2,0,1)
            blend_depths = None
            blend_normals = None
        blend_alphas = rast_alphas[0].permute(2,0,1)
        
        # ADDED ALIGNMENT TO PREVENT NANS
        blend_colors = blend_colors.clamp(0, 1)
        blend_alphas = blend_alphas.clamp(0, 1)
        if blend_depths is not None:
            blend_depths = blend_depths.div(blend_alphas.add(1e-6))

        blend_semantics = None
        return blend_colors, blend_alphas, blend_semantics, blend_depths, blend_normals, info


def add_safety_gaussians(sources_gaussians, extrinsic, intrinsic, batch_idx, depth=10.0):
    """
    Computes a safety gaussian positioned far in the center of the target view 
    and concatenates it to the sources_gaussians.
    """
    device = sources_gaussians['opacities'].device
    dtype = sources_gaussians['opacities'].dtype

    # 1. Define the point in Camera Space (0, 0, z)
    # (0,0) in camera space corresponds to the principal point (center of image)
    # We use homogenous coordinates [x, y, z, 1]
    p_cam = torch.tensor([0.0, 0.0, depth, 1.0], device=device, dtype=dtype)

    # 2. Convert Camera Space -> World Space
    # Extrinsic is typically World-to-Camera (W2C). We need the inverse (C2W).
    c2w = closed_form_inverse_se3(extrinsic[None])[0]
    p_world = (c2w @ p_cam)[:3]

    # 3. Create and Append the Safety Gaussian
    for key in sources_gaussians.keys():
        current_data = sources_gaussians[key].detach()
        # Get feature dimension (e.g., 3 for means, 4 for quats, etc.)
        dim = current_data.shape[1] if current_data.dim() > 1 else 1
        
        # Create a new placeholder tensor for 1 gaussian
        new_primitive = torch.zeros((1, *current_data.shape[1:]), device=device, dtype=dtype)

        if key in ['means3d', 'xyz', 'positions']:
            new_primitive[0] = p_world
        elif key == 'opacities':
            new_primitive[0] = 1.0  # Fully opaque to ensure rasterizer catches it
        elif key == 'scales':
            new_primitive[:] = 0.1 # Small finite scale to avoid singular covariance
        elif key == 'rotations':
            # Identity Quaternion (w, x, y, z) -> [1, 0, 0, 0]
            new_primitive[0, 0] = 1.0 
        elif key == 'conf': # Confidence map support
            new_primitive[0] = 1.0
        # For 'features' or 'sh', we leave them as 0.0 (Black)

        # Concatenate
        sources_gaussians[key] = torch.cat([current_data, new_primitive], dim=0)

    return sources_gaussians

def save_ply(path, gaussians_attr):
    num_gaussians = gaussians_attr['means'].shape[0]
    means = gaussians_attr['means'].detach().cpu().float().numpy()
    normals = np.zeros_like(means)  # todo unused value, only for SIBR viewer
    features_dc = gaussians_attr['features_dc'].detach()  # n, 1, 3
    features_sh = gaussians_attr['features_sh'].detach()  # n, k, 3
    if features_sh.shape[1] == 0:
        sh_degree = 0
    else:
        sh_degree = int(math.sqrt(features_sh.shape[-2] + 1)) - 1
    
    features_dc = features_dc.reshape(num_gaussians, 3).contiguous().cpu().float().numpy()
    features_sh = features_sh.mT.reshape(num_gaussians, -1).contiguous().cpu().float().numpy()
    opacitys = gaussians_attr['opacities'].detach().cpu().float().numpy()
    scale = gaussians_attr['scales'].detach().cpu().float().numpy()
    quats = gaussians_attr['quats'].detach().cpu().float().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(sh_degree)]
    attributes = np.concatenate([means, normals, features_dc, features_sh, opacitys, scale, quats], axis=1)
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def construct_list_of_attributes(degree=None):
    if degree is None:
        degree = 0
    att_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    att_list += ['f_dc_0', 'f_dc_1', 'f_dc_2']
    num_rest = degree * (degree + 2) * 3
    att_list += [f'f_rest_{i}' for i in range(num_rest)]
    att_list += ['opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    return att_list

def quaternion_multiply(q1, q2):
    """
    Multiplies two tensors of quaternions.

    Args:
        q1 (torch.Tensor): A tensor of quaternions with shape (..., 4).
        q2 (torch.Tensor): A tensor of quaternions with shape (..., 4).

    Returns:
        torch.Tensor: The product of the two quaternion tensors with shape (..., 4).
    """
    # Extract real and imaginary parts
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    # Hamilton product formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def build_rotation(r):
    #norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    #q = r / norm[:, None]
    q = F.normalize(r,dim=1)
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def extract_rotation(R):
    N = R.shape[0]
    q = torch.zeros((N,4),dtype=R.dtype).to(R.device)

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    ids1 = tr > 0
    ids2 = (~ids1) & ((R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]))
    ids3 = (~ids1) & (~ids2) & (R[:, 1, 1] > R[:, 2, 2])
    ids4 = (~ids1) & (~ids2) & (~ids3)

    S = torch.sqrt(tr[ids1] + 1.0) * 2
    qw = 0.25 * S;
    qx = (R[ids1, 2, 1] - R[ids1, 1, 2]) / S
    qy = (R[ids1, 0, 2] - R[ids1, 2, 0]) / S
    qz = (R[ids1, 1, 0] - R[ids1, 0, 1]) / S
    q[ids1] = torch.vstack((qw,qx,qy,qz)).T
    #q[ids1] = torch.vstack((qx,qy,qz,qw)).T

    S = torch.sqrt(1.0 + R[ids2, 0, 0] - R[ids2, 1, 1] - R[ids2, 2, 2]) * 2
    qw = (R[ids2, 2, 1] - R[ids2, 1, 2]) / S
    qx = 0.25 * S
    qy = (R[ids2, 0, 1] + R[ids2, 1, 0]) / S
    qz = (R[ids2, 0, 2] + R[ids2, 2, 0]) / S
    q[ids2] = torch.vstack((qw,qx,qy,qz)).T
    #q[ids2] = torch.vstack((qx,qy,qz,qw)).T

    S = torch.sqrt(1.0 + R[ids3, 1, 1] - R[ids3, 0, 0] - R[ids3, 2, 2]) * 2
    qw = (R[ids3, 0, 2] - R[ids3, 2, 0]) / S
    qx = (R[ids3, 0, 1] + R[ids3, 1, 0]) / S
    qy = 0.25 * S;
    qz = (R[ids3, 1, 2] + R[ids3, 2, 1]) / S
    q[ids3] = torch.vstack((qw,qx,qy,qz)).T
    #q[ids3] = torch.vstack((qx,qy,qz,qw)).T

    S = torch.sqrt(1.0 + R[ids4, 2, 2] - R[ids4, 0, 0] - R[ids4, 1, 1]) * 2
    qw = (R[ids4, 1, 0] - R[ids4, 0, 1]) / S
    qx = (R[ids4, 0, 2] + R[ids4, 2, 0]) / S
    qy = (R[ids4, 1, 2] + R[ids4, 2, 1]) / S
    qz = 0.25 * S
    q[ids4] = torch.vstack((qw,qx,qy,qz)).T
    #q[ids4] = torch.vstack((qx,qy,qz,qw)).T

    norm = torch.sqrt(q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + q[:,3]*q[:,3])
    q = q / norm[:, None]
    return q


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = torch.nn.funtionalconv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.funtionalconv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.funtionalconv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.funtionalconv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.funtionalconv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def inverse_sigmoid(x, epsilon=0.0):
    return torch.log(epsilon+(x/(1-x+epsilon)))

def depth_tv_loss(pred_depth):
    pred_depth = pred_depth[...,0]
    num_gaussians = pred_depth.shape[-1]
    perm = torch.randperm(num_gaussians)
    target_depth  = pred_depth[...,perm]

    return l1_loss(pred_depth,target_depth)


def get_pcd_base_TORCH(H, W, u0, v0, fx, fy):
    H = int(H)
    W = int(W)
    u0 = float(u0)
    v0 = float(v0)
    fx = float(fx)
    fy = float(fy)
    x_row = torch.arange(0, W, device='cuda')
    x = x_row.repeat(H, 1)
    u_m_u0 = x - u0

    y_col = torch.arange(0, H, device='cuda')
    y = y_col.repeat(W, 1).T
    v_m_v0 = y - v0

    x = u_m_u0 / fx
    y = v_m_v0 / fy
    z = torch.ones_like(x, device='cuda')
    pw = torch.stack([x, y, z], dim=0)
    return pw

def bilinear_interpolation(images: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Performs bilinear interpolation on a batch of RGB images to get color values at given points.

    Args:
        images (torch.Tensor): A batch of N RGB images of shape (N, C, H, W), where C=3 for RGB.
        points (torch.Tensor): A batch of D points for each image, with shape (N, D, 2).
                             Each point is represented by its (x, y) floating pixel coordinates.
                             x corresponds to the width dimension (W), and y corresponds to the height dimension (H).
                             The coordinates are expected to be within the range [0, W-1] and [0, H-1] respectively.

    Returns:
        torch.Tensor: A tensor of shape (N, D, C) containing the interpolated RGB color for each point.
    """
    N, C, H, W = images.shape
    N_points, D, _ = points.shape
    

    assert N == N_points, "Number of images and point batches must be the same."
    assert _ == 2, "Points must have shape (N, D, 2) with (x, y) coordinates."

    # Normalize pixel coordinates to the range [-1, 1] for grid_sample
    normalized_points = points.clone()
    normalized_points[..., 0] = 2 * normalized_points[..., 0] / (W) - 1
    normalized_points[..., 1] = 2 * normalized_points[..., 1] / (H) - 1

    # Create a grid for grid_sample. The shape should be (N, D, 1, 2)
    grid = normalized_points.unsqueeze(2)

    # Use grid_sample for bilinear interpolation
    interpolated_colors = F.grid_sample(images, grid, mode='bilinear', padding_mode='border', align_corners=False)

    # The output of grid_sample will have shape (N, C, D, 1).
    # We need to reshape it to (N, D, C).
    interpolated_colors = interpolated_colors.permute(0, 2, 3, 1).squeeze(2)

    return interpolated_colors

def RGB2SH(rgb):
    return (rgb - 0.5) / 0.28209479177387814

def SH2RGB(sh):
    return (sh * 0.28209479177387814) + 0.5

def color_from_images(images, pixel_coords):
    N, C, H, W = images.shape
    N, h, w, G, D = pixel_coords.shape
    patch_size = H // h

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = torch.from_numpy(u)[None,:,:,None].expand(-1,-1,-1,G).to(pixel_coords.device) + 0.5
    v = torch.from_numpy(v)[None,:,:,None].expand(-1,-1,-1,G).to(pixel_coords.device) + 0.5
    u = (u + pixel_coords[...,0] * 0.5) * patch_size
    v = (v + pixel_coords[...,1] * 0.5) * patch_size

    pixel_coords = rearrange(torch.stack([u,v],dim=-1),'N H W G D -> N (H W G) D')
    rgb = bilinear_interpolation(images, pixel_coords)
    sh = RGB2SH(rgb)
    return sh

def patchify_images(images, patch_size=(3, 3), stride=(1, 1)):
    """
    Patchifies a batch of images.
    
    Args:
        images (Tensor): A tensor of shape (B, D, H, W), where
                         B = batch size, D = channels, H = height, W = width.
        patch_size (tuple): Size of the patches (h, w).
        stride (tuple): Stride of the sliding window (h_stride, w_stride).
        
    Returns:
        patches (Tensor): A tensor of shape (B, D, P, h, w), where P is the number of patches.
    """
    B, D, H, W = images.shape
    h, w = patch_size
    h_stride, w_stride = stride
    
    # Use unfold to extract patches
    unfolded = images.unfold(2, h, h_stride).unfold(3, w, w_stride)  # (B, D, P_h, P_w, h, w)
    
    # Reshape to (B, D, P, h, w)
    patches = unfolded.contiguous().view(B, D, -1, h, w)  # P will be flattened into the 3rd dimension
    
    return patches

def patchify(images: torch.Tensor, patch_size: tuple[int, int]) -> torch.Tensor:
    """
    Patchifies a batch of images into non-overlapping patches.

    Args:
        images (torch.Tensor): A tensor of shape (B, D, H, W) representing a batch
                               of B images with D channels, height H, and width W.
        patch_size (tuple[int, int]): A tuple (h, w) specifying the height and
                                      width of each patch.

    Returns:
        torch.Tensor: A tensor of shape (B, P_h * P_w, D, h, w) where:
                      - B is the batch size.
                      - P_h is the number of patches along the height dimension (H // h).
                      - P_w is the number of patches along the width dimension (W // w).
                      - D is the number of channels.
                      - h is the patch height.
                      - w is the patch width.
    """
    B, D, H, W = images.shape
    h, w = patch_size

    assert H % h == 0, f"Image height ({H}) must be divisible by patch height ({h})."
    assert W % w == 0, f"Image width ({W}) must be divisible by patch width ({w})."

    num_patches_h = H // h
    num_patches_w = W // w
    num_patches = num_patches_h * num_patches_w

    # Use unfold to extract patches
    patches = images.unfold(2, h, h).unfold(3, w, w)

    # Reshape the patches to the desired output shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, num_patches, D, h, w)

    return patches

class NormalSupervisionLoss(torch.nn.Module):
    def __init__(self, sobel_kernel_size= (5, 5)):
        super(NormalSupervisionLoss, self).__init__()

        self.sobel_kernel_size = sobel_kernel_size
        self.sobel_x = torch.from_numpy(custom_sobel(sobel_kernel_size, 0)).view(1, 1, *sobel_kernel_size).expand(3, 1, -1, -1).float()
        self.sobel_y = torch.from_numpy(custom_sobel(sobel_kernel_size, 1)).view(1, 1, *sobel_kernel_size).expand(3, 1, -1, -1).float()

    def compute_normal_from_depth_cam(self, depth, intr, H, W):
        pcd_base = get_pcd_base_TORCH(H, W, intr[0,2], intr[1,2], intr[0,0],intr[1,1])
        pcd = depth * pcd_base
        pcd_pad = F.pad(pcd[None], (self.sobel_kernel_size[1]//2, self.sobel_kernel_size[1]//2, self.sobel_kernel_size[0]//2, self.sobel_kernel_size[0]//2), "replicate")
        dx_cuda = F.conv2d(pcd_pad, self.sobel_x.to(depth.device), stride=1, padding=0, groups=3).squeeze()
        dy_cuda = F.conv2d(pcd_pad, self.sobel_y.to(depth.device), stride=1, padding=0, groups=3).squeeze()
        normal_from_depth = - torch.cross(dx_cuda, dy_cuda, dim=0) # looking towards camera center
        normal_from_depth = F.normalize(normal_from_depth, p=2, dim=0)
        return normal_from_depth
    
    def compute_normal_from_depth_world(self, depth, intr, w2c, H, W):
        """
        depth: (1, H, W) or (H, W)
        intr: (3, 3) or (4, 4) intrinsics
        w2c: (4, 4) world-to-camera extrinsics
        """
        # 1. Reconstruct Point Cloud in Camera Space
        pcd_base = get_pcd_base_TORCH(H, W, intr[0,2], intr[1,2], intr[0,0], intr[1,1])
        pcd_cam = depth * pcd_base  # Shape: (3, H, W)

        # 2. Apply Sobel Filter to get gradients (in Camera Space)
        # Pad to keep dimensions consistent
        pad_h = self.sobel_kernel_size[0] // 2
        pad_w = self.sobel_kernel_size[1] // 2
        pcd_pad = F.pad(pcd_cam.unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), "replicate")

        # Convolve. Input (1, 3, H+pad, W+pad) -> Output (1, 3, H, W)
        dx = F.conv2d(pcd_pad, self.sobel_x.to(depth.device), stride=1, padding=0, groups=3)
        dy = F.conv2d(pcd_pad, self.sobel_y.to(depth.device), stride=1, padding=0, groups=3)
        
        # Remove batch dim for cross product: (3, H, W)
        dx = dx.squeeze(0) 
        dy = dy.squeeze(0)

        # 3. Compute Normals in Camera Space
        # Note: -cross(x,y) usually points towards the camera in OpenCV coordinates
        normal_cam = -torch.cross(dx, dy, dim=0) 
        normal_cam = F.normalize(normal_cam, p=2, dim=0) # (3, H, W)

        # 4. Transform Normals to World Space
        # N_world = R_w2c^T @ N_cam
        # w2c maps World -> Camera, so Rotation part maps V_world -> V_cam
        # We need the inverse (transpose) to go V_cam -> V_world
        
        R_w2c = w2c[:3, :3] # (3, 3)
        R_c2w = R_w2c.permute(1, 0) # Transpose implies inverse for rotation matrices

        # Reshape normal_cam to (3, H*W) for matrix multiplication
        normal_cam_flat = normal_cam.view(3, -1)
        
        # Perform rotation: (3,3) @ (3, H*W) -> (3, H*W)
        normal_world_flat = torch.matmul(R_c2w, normal_cam_flat)
        
        # Reshape back to (3, H, W)
        normal_world = normal_world_flat.view(3, H, W)

        return normal_world

    def forward_noalign(self, rendered_normals, target_normals, mask=None):
        if mask is not None:
            lp_normal = (torch.abs(rendered_normals - target_normals) * mask).mean()
            cos_normal = ((1 - (rendered_normals * target_normals).sum(dim=1)) * mask).mean()
        else:
            lp_normal = torch.abs(rendered_normals - target_normals).mean()
            cos_normal = (1 - (rendered_normals * target_normals).sum(dim=1)).mean()
        return 0.01 * lp_normal + 0.01 * cos_normal

    def forward(self, rendered_normals, target_normals, mask=None):
        # 1. Check alignment: Dot product of Rendered vs Target
        # If dot < 0, they are opposite. We get the sign (-1 or +1).
        alignment = torch.sign((rendered_normals * target_normals).sum(dim=1, keepdim=True))
        
        # 2. Flip rendered normals to match the hemisphere of the target
        # If alignment is -1, this flips rendered_normal. If +1, it stays same.
        aligned_rendered_normals = rendered_normals * alignment

        # 3. Compute losses using the aligned normals
        if mask is not None:
            lp_normal = (torch.abs(aligned_rendered_normals - target_normals) * mask).mean()
            # Cosine distance is now safe to strictly maximize
            cos_normal = ((1 - (aligned_rendered_normals * target_normals).sum(dim=1)) * mask).mean()
        else:
            lp_normal = torch.abs(aligned_rendered_normals - target_normals).mean()
            cos_normal = (1 - (aligned_rendered_normals * target_normals).sum(dim=1)).mean()

        return 0.01 * lp_normal + 0.01 * cos_normal
    

def custom_sobel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0]) 
           for i in range(shape[1]) 
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k


def psnr_pixels(img1, img2):
    mse = (((img1 - img2)) ** 2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_camera_poses(extrinsics, intrinsics, paths, file_path):
    num_frames = len(paths)
    cam_data = {}
    fx = torch.mean(intrinsics[:,0,0]).detach().cpu().item()
    fy = torch.mean(intrinsics[:,1,1]).detach().cpu().item()
    cam_data["w"]= 907#int(2.39 * 518)#
    cam_data["h"]= 518
    cam_data["fl_x"]= fx
    cam_data["fl_y"]= fy
    cam_data["cx"]= 907//2#int(2.39 * 518 * 0.5)
    cam_data["cy"]= 518//2
    cam_data["k1"]= 0.0
    cam_data["k2"]= 0.0
    cam_data["p1"]= 0.0
    cam_data["p2"]= 0.0
    cam_data["camera_model"]= "OPENCV"
    cam_data["frames"] = []
    for i in range(num_frames):
        frame_dict = {}
        path = paths[i].split('/')[-1]
        #frame_dict["file_path"] = 'images/' + path
        
        frame_dict["file_path"] = f'images_4/frame_{str(i).zfill(5)}.png'
        frame_dict["colmap_id"] = int(path[-7:-4])
        matrix = extrinsics[i].tolist()
        frame_dict["transform_matrix"] = matrix
        cam_data["frames"].append(frame_dict)
        

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cam_data, f, ensure_ascii=False, indent=2)

