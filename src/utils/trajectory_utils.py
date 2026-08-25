import torch
import numpy as np
from src.utils.geometry_utils import closed_form_inverse_se3
from scipy.spatial.transform import Slerp, Rotation
import matplotlib.pyplot as plt


def render_trajectory(model, gaussians, trajectory, intrinsics, num_video_frames=100, H=518, W=518, bg_color=torch.tensor([0.,0.,0.]),upscale_ratio=1.0):
    if upscale_ratio!= 1.0:
        H = int(H*upscale_ratio)
        W = int(W*upscale_ratio)
        intrinsics[:,:2] *= upscale_ratio
    with torch.no_grad():
        video_tensor = torch.zeros(num_video_frames,3,H,W)
        depth_tensor = torch.zeros(num_video_frames,3,H,W)
        for i in range(num_video_frames):
            render, depth, normals, alphas = model.render_image(gaussians,trajectory[i].float().cuda(),intrinsics[i].float().cuda(),H,W, 
                                                    conf_fusion=True,batch_idx=0, bg_color=bg_color)
            if i == 0:
                vmin = np.min(depth.cpu().numpy())
                vmax = np.max(depth.cpu().numpy())
            if render is not None:
                #render[:,torch.sum(render,dim=0)==0] = bg_color #rasterizer does not apply bg color correctly, override black patches
                video_tensor[i] = render.cpu()
                
                depth = (depth[0].detach().cpu().numpy() - vmin) / vmax
                depth = plt.cm.magma_r(depth)[..., :3]
                depth = torch.from_numpy(depth).permute(2, 0, 1).float()
                depth_tensor[i] = depth
    return video_tensor, depth_tensor

def generate_trajectory(num_frames, trajectory_mode, extrinsics, intrinsics, depth=None, time_warp_mode='linear'):
    """
    extrinsics: torch tensor (N, 4, 4) w2c matrices.
    intrinsics: torch tensor (3, 3) camera intrinsics matrix (K).
    depth: torch tensor (H, W) depth map of the first image.
    time_warp_mode: str, controls timing ('linear', 'ease_in', 'ease_out', 'smoothstep', 'cos_ease').
    
    Returns:
    traj_w2c: torch tensor (num_frames, 4, 4)
    traj_K: torch tensor (num_frames, 3, 3)
    """
    
    # --- 1. Standard Interpolation Modes ---
    if trajectory_mode == 'interpolation':
        # Assuming interpolate_extrinsics handles just poses
        traj_w2c = interpolate_extrinsics(extrinsics[:,:3].detach().cpu(), num_frames)
        traj_K = intrinsics.unsqueeze(0).repeat(num_frames, 1, 1) # Constant K
        return traj_w2c, traj_K

    elif 'spiral' in trajectory_mode:
        # Placeholder for existing spiral logic integration
        # (Assuming you would adapt spiral to return intrinsics too, here we return constant K)
        if trajectory_mode == 'spiral':
            traj_np = generate_spiral_path(extrinsics[0,:3].detach().cpu().numpy(), N=num_frames, radius=0.1)
        else: # interpolation+spiral
            interp = interpolate_extrinsics(extrinsics[:,:3].detach().cpu(), num_frames//2)
            extrap = generate_spiral_path(interp[num_frames//4].cpu().numpy(), N=num_frames//2, radius=0.1)
            traj_np = np.concatenate([interp[:num_frames//4].numpy(), extrap, interp[num_frames//4:].numpy()], axis=0)
            
        traj_w2c = torch.from_numpy(traj_np).float().to(extrinsics.device)
        traj_K = intrinsics.unsqueeze(0).repeat(num_frames, 1, 1)
        return traj_w2c, traj_K

    # --- 2. Bullet Time Modes ---
    if depth is None:
        raise ValueError("Depth map is required for bullet time effects.")

    pivot, axes, start_c2w = get_pivot_and_axes(extrinsics, depth)

    # Dictionary mapping mode names to functions
    # All functions signature: (num_frames, pivot, axes, start_c2w, intrinsics, time_mode)
    bullet_modes = {
        'bullet_time_linear':      lambda: generate_bullet_time_linear(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode, angle=40),
        'bullet_time_circle':      lambda: generate_bullet_time_circle(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode),
        'bullet_time_loop':        lambda: generate_bullet_time_loop(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode, max_angle=40),
        'bullet_time_linear_zoom': lambda: generate_bullet_time_linear_zoom(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode, angle=-60, zoom_factor=0.4),
        'bullet_time_dolly_zoom':  lambda: generate_bullet_time_dolly_zoom(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode, zoom_ratio=2.0),
        'bullet_time_sway':        lambda: generate_bullet_time_sway(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode),
        'bullet_time_wobble':      lambda: generate_bullet_time_wobble(num_frames, pivot, axes, start_c2w, intrinsics, time_warp_mode),
    }

    if trajectory_mode in bullet_modes:
        return bullet_modes[trajectory_mode]()
    
    raise NotImplementedError(f"Mode {trajectory_mode} not implemented")


def interpolate_extrinsics(extrinsics, num_frames):
    """
    Interpolates a smooth camera trajectory from a tensor of extrinsic transformation matrices
    and returns the interpolated poses as a torch tensor of extrinsic matrices.

    Args:
        extrinsics (torch.Tensor): A tensor of shape (N, 3, 4) representing the extrinsic
            transformation matrices of N cameras. Each matrix transforms points from
            world coordinates to camera coordinates.
        num_frames (int): The desired number of frames in the output camera trajectory.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, 3, 4) representing the interpolated
        camera poses as extrinsic transformation matrices.
    """
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError("Input 'extrinsics' must be a tensor of shape (N, 3, 4).")

    num_key_frames = extrinsics.shape[0]
    if num_frames < num_key_frames:
        raise ValueError("Number of output frames should be greater than or equal to the number of input camera poses.")

    # Extract translations and rotations from the extrinsic matrices
    translations = extrinsics[:, :3, 3].numpy()
    rotation_matrices = extrinsics[:, :3, :3].numpy()
    rotations = Rotation.from_matrix(rotation_matrices)

    # Define interpolation times (normalized between 0 and 1)
    key_frame_times = np.linspace(0, 1, num_key_frames)
    interpolation_times = np.linspace(0, 1, num_frames)

    # Interpolate translations linearly
    interpolated_translations = np.zeros((num_frames, 3))
    for i in range(3):
        interpolated_translations[:, i] = np.interp(interpolation_times, key_frame_times, translations[:, i])

    # Interpolate rotations using spherical linear interpolation (Slerp)
    slerp = Slerp(key_frame_times, rotations)  # Pass the list of Rotation objects
    interpolated_rotations = slerp(interpolation_times) # Get interpolated Rotation objects
    interpolated_rotations_matrix = np.stack([r.as_matrix() for r in interpolated_rotations])

    # Construct the output extrinsic matrices
    interpolated_extrinsics = np.concatenate([interpolated_rotations_matrix, interpolated_translations[:, :, None]], axis=-1)

    return torch.from_numpy(interpolated_extrinsics).float()

def generate_spiral_path(input_pose, N, radius):

    #poses = np.tile(poses_avg(input_poses)[None,:,:], (N,1,1))
    poses = np.tile(input_pose, (N,1,1))
    R = poses[...,:3]
    t = poses[...,3:]

    # spiral path
    theta = np.linspace(0,2*np.pi,N)
    r = np.concatenate([np.linspace(0,radius, N//4),radius*np.ones(N//2),np.linspace(radius,0, N//4)])
    tx = r * np.cos(theta)
    ty = r * np.sin(theta)
    tz = np.zeros_like(theta)

    trans = R @ np.stack([tx,ty,tz], axis=1)[..., None]
    t += trans

    lookdir = poses[...,2:3] #- 50*trans
    
    out_poses = np.zeros((N,3,4))
    for i in range(N):
        out_poses[i] = viewmatrix(lookdir[i].flatten(), input_pose[:,1], t[i].flatten())

    out_poses = np.concatenate([R,t], axis=-1)

    return out_poses

def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def get_time_vector(num_frames, device, mode='linear'):
    """Generates a time vector t in [0, 1] with specific warping."""
    t = torch.linspace(0, 1, num_frames, device=device)
    
    if mode == 'linear':
        return t
    elif mode == 'ease_in':
        return t ** 2
    elif mode == 'ease_out':
        return 1.0 - (1.0 - t) ** 2
    elif mode == 'smoothstep':
        return 3 * t**2 - 2 * t**3
    elif mode == 'cos_ease':
        # Slower start/end, faster middle (cosine curve)
        return 0.5 * (1 - torch.cos(t * np.pi))
    
    return t # Default to linear

def compute_trajectory_from_params(pivot, start_c2w, start_K, axes, angle_vals, elev_vals, radius_scales, focal_scales):
    """
    Reconstructs Poses and Intrinsics from spherical coordinates relative to pivot.
    """
    device = start_c2w.device
    up_axis, right_axis = axes
    start_pos = start_c2w[:3, 3]
    base_radius_vec = start_pos - pivot
    
    traj_c2w = []
    traj_K = []

    for theta_deg, phi_deg, r_scale, f_scale in zip(angle_vals, elev_vals, radius_scales, focal_scales):
        # 1. Calculate Position
        theta = torch.deg2rad(theta_deg)
        phi = torch.deg2rad(phi_deg)

        # Rotation Matrices (Rodrigues)
        I = torch.eye(3, device=device)
        
        # Azimuth (around Up)
        K_az = torch.tensor([[0, -up_axis[2], up_axis[1]], [up_axis[2], 0, -up_axis[0]], [-up_axis[1], up_axis[0], 0]], device=device)
        R_az = I + torch.sin(theta) * K_az + (1 - torch.cos(theta)) * (K_az @ K_az)
        
        # Elevation (around Right)
        K_el = torch.tensor([[0, -right_axis[2], right_axis[1]], [right_axis[2], 0, -right_axis[0]], [-right_axis[1], right_axis[0], 0]], device=device)
        R_el = I + torch.sin(phi) * K_el + (1 - torch.cos(phi)) * (K_el @ K_el)
        
        # Apply Rotations and Scale to Radius Vector
        current_vec = R_az @ (R_el @ base_radius_vec)
        current_vec = current_vec * r_scale
        
        new_pos = pivot + current_vec
        
        # 2. Calculate Orientation (LookAt)
        # Using linalg.cross as requested
        z_axis = new_pos - pivot # Actually Pivot - Pos is forward? 
        # LookAt: Z points from Camera to Object (standard) or Object to Camera?
        # Standard Colmap/OpenCV: Z is positive forward. 
        # So Z = Pivot - Camera (vector pointing to object).
        # Wait, if Z is forward depth, Z should point TO the scene.
        z_axis = pivot - new_pos 
        z_axis = z_axis / torch.norm(z_axis)
        
        x_axis = torch.linalg.cross(up_axis, z_axis) 
        x_axis = x_axis / torch.norm(x_axis)
        
        y_axis = torch.linalg.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        
        # R_new = [x, y, z] columns
        new_rot = torch.stack([x_axis, y_axis, z_axis], dim=1)
        
        new_c2w = torch.eye(4, device=device)
        new_c2w[:3, :3] = new_rot
        new_c2w[:3, 3] = new_pos
        traj_c2w.append(new_c2w)
        
        # 3. Calculate Intrinsics (Zoom)
        new_K = start_K.clone()
        new_K[0, 0] *= f_scale # fx
        new_K[1, 1] *= f_scale # fy
        traj_K.append(new_K)

    # Stack and invert C2W to W2C
    traj_w2c = closed_form_inverse_se3(torch.stack(traj_c2w))
    traj_K_stack = torch.stack(traj_K)
    
    return traj_w2c, traj_K_stack

# ---------------------------------------------------------------------------------
# Trajectory Mode Implementations
# ---------------------------------------------------------------------------------

def generate_bullet_time_linear(num_frames, pivot, axes, start_c2w, K, time_mode, angle=40):
    device = start_c2w.device
    t = get_time_vector(num_frames, device, time_mode)
    
    angles = t * angle
    elevs = torch.zeros_like(t)
    radii = torch.ones_like(t)
    focals = torch.ones_like(t)
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

def generate_bullet_time_circle(num_frames, pivot, axes, start_c2w, K, time_mode):
    device = start_c2w.device
    t = get_time_vector(num_frames, device, time_mode)
    
    angles = t * 360.0
    elevs = torch.zeros_like(t)
    radii = torch.ones_like(t)
    focals = torch.ones_like(t)
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

def generate_bullet_time_loop(num_frames, pivot, axes, start_c2w, K, time_mode, max_angle=40):
    device = start_c2w.device
    # For loop, we want linear time for the parameterization t to complete the cycle,
    # but we can warp the underlying angle calculation if desired.
    # Usually loops are better with linear 't' inputs to sin/cos, or cos_ease.
    t_raw = torch.linspace(0, 1, num_frames, device=device) 
    t_rad = t_raw * 2 * np.pi
    
    # Warping the amplitude progress if time_mode is not linear might look weird on a loop,
    # so we apply warping to the t_raw to speed up/slow down the loop execution itself.
    t_warped = get_time_vector(num_frames, device, time_mode) * 2 * np.pi
    
    angles = max_angle * (1 - torch.cos(t_warped)) / 2.0
    elevs = (max_angle / 6.0) * torch.sin(t_warped)
    radii = torch.ones_like(t_raw)
    focals = torch.ones_like(t_raw)
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

def generate_bullet_time_linear_zoom(num_frames, pivot, axes, start_c2w, K, time_mode, angle=40, zoom_factor=0.6):
    device = start_c2w.device
    t = get_time_vector(num_frames, device, time_mode)
    
    angles = t * angle
    elevs = torch.zeros_like(t)
    # Lerp radius
    radii = 1.0 + t * (zoom_factor - 1.0)
    focals = torch.ones_like(t)
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

# --- NEW MODES ---

def generate_bullet_time_dolly_zoom(num_frames, pivot, axes, start_c2w, K, time_mode, zoom_ratio=2.0):
    """
    Vertigo Effect: Moves camera backwards while zooming in (increasing focal length).
    zoom_ratio > 1.0 means moving further away.
    """
    device = start_c2w.device
    t = get_time_vector(num_frames, device, time_mode)
    
    angles = torch.zeros_like(t)
    elevs = torch.zeros_like(t)
    
    # Move camera from dist d to dist d*zoom_ratio
    radii = 1.0 + t * (zoom_ratio - 1.0)
    
    # To keep object size constant: f1/d1 = f2/d2 -> f2 = f1 * (d2/d1)
    # Here d2/d1 is exactly the radius scale
    focals = radii 
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

def generate_bullet_time_sway(num_frames, pivot, axes, start_c2w, K, time_mode):
    """
    Idle animation: Gentle figure-8 sway around the center.
    """
    device = start_c2w.device
    # Linear t for continuous oscillation
    t = torch.linspace(0, 1, num_frames, device=device)
    # Use time_mode to ease in the start of the sway? 
    # Or just use linear to ensure it loops perfectly if needed.
    # Let's assume a single sway back and forth.
    
    cycle = 2 * np.pi * t
    
    # Sway left/right
    angles = 10.0 * torch.sin(cycle) 
    # Bob up/down at half speed (figure 8)
    elevs = 5.0 * torch.sin(cycle / 2.0)
    
    radii = torch.ones_like(t)
    focals = torch.ones_like(t)
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

def generate_bullet_time_wobble(num_frames, pivot, axes, start_c2w, K, time_mode):
    """
    Hand-held camcorder feeling: organic noise-like movement + slight zoom instability.
    """
    device = start_c2w.device
    t = get_time_vector(num_frames, device, time_mode)
    
    # Lissajous-like chaotic curves
    # Prime number multipliers help avoid repeating patterns quickly
    angles = 5.0 * torch.sin(t * 3.0) + 2.0 * torch.cos(t * 7.0)
    elevs  = 3.0 * torch.cos(t * 4.0) + 1.5 * torch.sin(t * 5.0)
    
    # Slight "breathing" in radius
    radii = 1.0 + 0.05 * torch.sin(t * 6.0)
    
    # Slight instability in zoom (focus breathing)
    focals = 1.0 + 0.02 * torch.cos(t * 5.0)
    
    return compute_trajectory_from_params(pivot, start_c2w, K, axes, angles, elevs, radii, focals)

# ---------------------------------------------------------------------------------
# Helpers for Setup
# ---------------------------------------------------------------------------------

def get_pivot_and_axes(extrinsics, depth):
    """Calculates pivot and axes. Same as before."""
    start_w2c = extrinsics[0]
    start_c2w = closed_form_inverse_se3(extrinsics)[0]
    device = start_w2c.device
    
    H, W = depth.shape
    center_depth = depth[H//2, W//2]
    
    pivot_cam = torch.tensor([0.0, 0.0, center_depth, 1.0], device=device)
    pivot_world = (start_c2w @ pivot_cam)[:3]

    up_axis = start_c2w[:3, 1] 
    up_axis = up_axis / torch.norm(up_axis)
    
    right_axis = start_c2w[:3, 0] 
    right_axis = right_axis / torch.norm(right_axis)

    return pivot_world, (up_axis, right_axis), start_c2w