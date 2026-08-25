import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
import roma
import copy

def l1_loss(pred: Tensor, gt: Tensor, mask: Tensor = None) -> Tensor:
    """
    Args:
        pred: 3, h, w
        gt: 3, h, w
        mask:  h, w
    Returns:
    """
    if mask is None:
        return torch.abs((pred - gt)).mean()
    else:
        mask_sum = mask.sum().add(1.0)
        return torch.sum(torch.abs(pred - gt) * mask, dim=[-1, -2]).div(mask_sum).mean()
        #return (torch.abs((pred - gt)) * mask).mean()

def geometry_consistency_loss(data, teacher_data, use_depth_conf=False, shared_intrinsics=True, normalize=False):

    loss = 0
    log_dict = {}
    
    num_images = data["depths"].shape[1]
    H, W = data["depths"].shape[-2:]
    
    if normalize:
        scene_scales = torch.mean(teacher_data["depths"],dim=(1,2,3,4)).detach()
    #depth loss
    pred_depths = data["depths"]
    target_depth = teacher_data["depths"]
    if normalize:
        pred_depths = pred_depths / scene_scales[:,None,None,None,None]
        target_depth = target_depth / scene_scales[:,None,None,None,None]

    if use_depth_conf:
        target_depth_conf = teacher_data["depth_conf"]
        depth_loss = l1_loss(pred_depths, target_depth, target_depth_conf) * 100
    else:
        depth_loss = l1_loss(pred_depths, target_depth)
    log_dict["supervision_losses/depth_loss"] = depth_loss.detach().cpu()
    loss += depth_loss #* 10
    
    #pointmap loss
    if "pointmaps" in data:
        pred_pointmaps = data["pointmaps"]
        target_pointmaps = teacher_data["pointmaps"]
        if use_depth_conf:
            target_conf = teacher_data["pointmaps_conf"]
            pointmaps_loss = l1_loss(pred_pointmaps, target_pointmaps, target_conf) * 100
        else:
            pointmaps_loss = l1_loss(pred_pointmaps, target_pointmaps)
        log_dict["supervision_losses/pointmaps_loss"] = pointmaps_loss.detach().cpu()
        loss += pointmaps_loss #* 10

    #intrinsics loss
    if "cam_params" in data:
        pred_fov = data["cam_params"][...,7:9]
        target_fov = teacher_data["cam_params"][...,7:9]
    else:
        pred_fov = K_to_fov(data["intrinsics"], H, W)
        target_fov = K_to_fov(teacher_data["intrinsics"], H, W)
    
    # 3. Stack the results to match the desired output shape (B, N, 2)
    
    intrinsics_loss = l1_loss(pred_fov, target_fov)
    log_dict["supervision_losses/intrinsics_loss"] = intrinsics_loss.detach().cpu()
    loss += intrinsics_loss * 10#0
    
    if num_images>1:
        if "cam_params" in data:
            pred_transl = data["cam_params"][..., :3]
            target_transl = teacher_data["cam_params"][..., :3]
        else:
            pred_transl = data["extrinsics"][:,:,:3,3]
            target_transl = teacher_data["extrinsics"][:,:,:3,3]
        if normalize:
            pred_transl = pred_transl / scene_scales[:,None,None]
            target_transl = target_transl / scene_scales[:,None,None]
        translation_loss = l1_loss(pred_transl, target_transl)
        log_dict["supervision_losses/translation_loss"] = translation_loss.detach().cpu()
        loss += translation_loss * 100
        
        pred_orient = data["extrinsics"][:,:,:3,:3]
        target_orient = teacher_data["extrinsics"][:,:,:3,:3]
        rotation_loss = roma.rotmat_geodesic_distance(pred_orient, target_orient).mean()
        log_dict["supervision_losses/rotation_loss"] = rotation_loss.detach().cpu()
        loss += rotation_loss * 10

        if shared_intrinsics:
            mean_intrinsics = torch.mean(pred_fov,dim=1,keepdim=True)
            shared_intrinsics_loss = l1_loss(pred_fov,mean_intrinsics)
            log_dict["regularization_losses/shared_intrinsics_loss"] = shared_intrinsics_loss.detach().cpu()
            loss += shared_intrinsics_loss * 100

    return loss, log_dict

def K_to_fov(K, H, W):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    fov_x = 2 * torch.atan(W / (2 * fx))
    fov_y = 2 * torch.atan(H / (2 * fy))
    fov = torch.stack([fov_x, fov_y], dim=-1)
    return fov

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100, replace_value=0):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        
        replace_tensor = torch.ones_like(input_tensor) * replace_value
        print(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            replace_tensor,
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor
    
class EMAHelper:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        if device:
            self.teacher.to(device)

    def update(self, student_model):
        """Exponential moving average update: theta_t = decay * theta_t + (1-decay) * theta_s"""
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), student_model.parameters()):
                t_param.data.mul_(self.decay).add_(s_param.data, alpha=(1 - self.decay))

