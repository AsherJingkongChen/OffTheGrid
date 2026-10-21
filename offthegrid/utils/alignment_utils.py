import torch
import numpy as np
from scipy.optimize import minimize


def get_camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    
    # .mT performs a matrix transpose on the last two dimensions
    cam_center = (-R.mT @ T[..., None])[..., 0]
    
    # If you wanted to strictly use the axis indices (assuming a 3D tensor):
    # cam_center = (-R.permute(0, 2, 1) @ T[..., None])[..., 0]
    
    return cam_center

def camera_pose_optimization_on_test_images(images, gaussians, extrinsics, intrinsics, decoder, num_iterations=100):
    B, num_images, _, H, W = images.shape
    for att in gaussians:
        gaussians[att] = gaussians[att].detach()
    images = images.detach()
    extrinsics = extrinsics.detach().clone()
    intrinsics = intrinsics.detach()

    use_conf = True

    l = []
    _orientation = matrix_to_quaternion(extrinsics[:,:,:3,:3])
    _translation = extrinsics[:,:,:3,3]
    _orientation = torch.nn.Parameter(_orientation.requires_grad_(True))
    _translation = torch.nn.Parameter(_translation.requires_grad_(True))
    l.append({'params': [_translation], 'lr': 0.005, "name": "translation"})
    l.append({'params': [_orientation], 'lr': 0.005, "name": "orientation"})
    optimizer = torch.optim.Adam(l, eps=1e-15)
    
    for k in range(num_iterations):
        
        quat = torch.nn.functional.normalize(_orientation,dim=-1)
        rot = quaternion_to_matrix(quat)
        extr = torch.cat([rot,_translation[...,None]],dim=-1)
        loss, _ = decoder.render_and_compute_losses(gaussians,
                                                    extr,
                                                    intrinsics,
                                                    images,
                                                    None,
                                                    use_depth_loss=False,
                                                    use_normals_loss=False,
                                                    use_opacity_loss=False,
                                                    use_milo=False,
                                                    self_render=False,
                                                    tto=False)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    
    quat = torch.nn.functional.normalize(_orientation,dim=-1)
    rot = quaternion_to_matrix(quat)
    extr = torch.cat([rot,_translation[...,None]],dim=-1)
    return extr[0].float()