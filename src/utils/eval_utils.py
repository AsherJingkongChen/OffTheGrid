import numpy as np
import torch
import torch.nn.functional as F
from vggt.utils.geometry import closed_form_inverse_se3
from scipy.optimize import minimize
from einops import rearrange
import matplotlib.pyplot as plt


def visualize_depth(images, pred_depth, gt_depth, max_depth=None, gt_normalization=False):

    B, N, C, H, W = images.shape
    images = rearrange(images,'B N C H W -> B C H (N W)').detach().cpu()
    
    pred_depth = rearrange(pred_depth,'B N C H W -> B C H (N W)').detach().cpu()
    gt_depth = rearrange(gt_depth,'B N C H W -> B C H (N W)').detach().cpu()
    final = []
    for b in range(images.shape[0]):
        if max_depth is None:
            max_depth = max(torch.max(gt_depth[b]),1e-3)
        gt = gt_depth[b,0] / max_depth
        #gt = torch.clamp(gt,min=0, max=1)
        colormap = plt.cm.viridis(gt.numpy())[..., :3]
        gt = torch.from_numpy(colormap).permute(2, 0, 1).float()

        if not gt_normalization:
            max_depth = max(torch.max(pred_depth[b]),1e-3)
        pred = pred_depth[b,0] / max_depth
        pred = torch.clamp(pred,min=0, max=1)
        colormap = plt.cm.viridis(pred.numpy())[..., :3]
        pred = torch.from_numpy(colormap).permute(2, 0, 1).float()
        final.append(torch.cat([images[b],pred,gt],dim=1))
    return final
    
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(
        torch.arange(N), 2, with_replacement=False
    ).unbind(-1)
    i1, i2 = [
        (i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]
    ]

    return i1, i2

def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = matrix_to_quaternion(rot_pred)
    q_gt = matrix_to_quaternion(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def camera_to_rel_deg(pred_cameras, gt_cameras, device, batch_size):
    """
    Calculate relative rotation and translation angles between predicted and ground truth cameras.

    Args:
    - pred_cameras: Predicted camera.
    - gt_cameras: Ground truth camera.
    - accelerator: The device for moving tensors to GPU or others.
    - batch_size: Number of data samples in one batch.

    Returns:
    - rel_rotation_angle_deg, rel_translation_angle_deg: Relative rotation and translation angles in degrees.
    """

    with torch.no_grad():
        # Convert cameras to 4x4 SE3 transformation matrices
        gt_se3 = gt_cameras.to(device)
        pred_se3 = pred_cameras.to(device)

        # Generate pairwise indices to compute relative poses
        pair_idx_i1, pair_idx_i2 = batched_all_pairs(
            batch_size, gt_se3.shape[0] // batch_size
        )
        pair_idx_i1 = pair_idx_i1.to(device)

        # Compute relative camera poses between pairs
        # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
        # This is possible because of SE3
        relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
            gt_se3[pair_idx_i2]
        )
        relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
            pred_se3[pair_idx_i2]
        )

        # Compute the difference in rotation and translation
        # between the ground truth and predicted relative camera poses
        rel_rangle_deg = rotation_angle(
            relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
        )
        rel_tangle_deg = translation_angle(
            relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
        )

    return rel_rangle_deg, rel_tangle_deg

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def calculate_auc(r_error, t_error, max_threshold=30, return_list=False):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Define histogram bins
    bins = torch.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(
        max_errors, bins=max_threshold + 1, min=0, max=max_threshold
    )

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    if return_list:
        return (
            torch.cumsum(normalized_histogram, dim=0).mean(),
            normalized_histogram,
        )
    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()

def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    
    initial_params = [s, t]  # s = 1, t = 0
    
    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))
    
    s, t = result.x  
    return s, t

def abs_relative_difference(output, target):
    abs_relative_diff = torch.abs(output.squeeze() - target.squeeze()) / target.squeeze()
    abs_relative_diff = torch.mean(abs_relative_diff)
    return abs_relative_diff

def squared_relative_difference(output, target):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    square_relative_diff = torch.mean(square_relative_diff)
    return square_relative_diff

def rmse_linear(output, target):
    actual_output = output.squeeze()
    actual_target = target.squeeze()
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.mean(diff2)
    rmse = torch.sqrt(mse)
    return rmse

def rmse_log(output, target):
    diff = torch.log(output) - torch.log(target)
    diff2 = torch.pow(diff, 2)
    mse = torch.mean(diff2)
    rmse = torch.sqrt(mse)
    return rmse

def silog_rmse(depth_pred, depth_gt):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    diff2 = torch.pow(diff, 2)

    first_term = torch.mean(diff2)
    second_term = torch.pow(torch.sum(diff), 2) / torch.numel(diff) / torch.numel(diff)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss

def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2 < threshold_val, one, zero)
    out_res = torch.mean(bit_mat)
    return out_res

def delta1_acc(pred, gt):
    return threshold_percentage(pred, gt, 1.25)

def delta2_acc(pred, gt):
    return threshold_percentage(pred, gt, 1.25**2)

def delta3_acc(pred, gt):
    return threshold_percentage(pred, gt, 1.25**3)

def i_rmse(output, target):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    diff2 = torch.pow(diff, 2)
    mse = torch.mean(diff2)
    rmse = torch.sqrt(mse)
    return rmse

def depth_eval(pred, gt, valid_depth=100, return_aligned=False, scene_alignment=False):
    
    # 1. Resize and squeeze
    if pred.shape != gt.shape:
        pred_resized = F.interpolate(pred, (gt.shape[2], gt.shape[3]), mode='bilinear')
    else:
        pred_resized = pred
    gt_squeezed = gt.squeeze(1) # Assuming N, 1, H, W -> N, H, W
    pred_squeezed = pred_resized.squeeze(1)

    metrics = {
        "AbsRel": abs_relative_difference,
        "SqRel": squared_relative_difference,
        "RMSElinear": rmse_linear,
        "RMSElog": rmse_log,
        "InvRMSE": i_rmse,
        "SILogRMSE": silog_rmse,
        "d1": delta1_acc, "d2": delta2_acc, "d3": delta3_acc,
    }

    out_dict = {}
    if return_aligned:
        depth_aligned = []

    # 2. OPTIONAL: Compute Global S, T (Scene Alignment)
    s, t = 1.0, 0.0
    if scene_alignment:
        # Create a mask for the whole scene/batch
        valid_mask_scene = (gt_squeezed > 1e-5) & (gt_squeezed < valid_depth)
        
        # Extract and clip valid pixels
        pred_scene = torch.clip(pred_squeezed[valid_mask_scene], min=1e-5, max=None)
        gt_scene = gt_squeezed[valid_mask_scene]

        if len(gt_scene) > 0:
            s, t = absolute_value_scaling(
                pred_scene, gt_scene,
                s=torch.median(gt_scene) / torch.median(pred_scene)
            )

    # 3. Iterate Frames
    for idx in range(gt.shape[0]):
        gt_frame = gt_squeezed[idx]
        pred_frame = pred_squeezed[idx]
        
        valid_mask = (gt_frame > 1e-5) & (gt_frame < valid_depth)
        
        pred_frame = torch.clip(pred_frame, min=1e-5, max=None)
        pred_depth_masked = pred_frame[valid_mask]
        gt_depth_masked = gt_frame[valid_mask]

        # If NOT scene alignment, compute s, t per frame
        if not scene_alignment:
            s, t = absolute_value_scaling(
                pred_depth_masked, gt_depth_masked,
                s=torch.median(gt_depth_masked) / torch.median(pred_depth_masked)
            )

        # Apply Scaling (Global or Local s,t)
        pred_aligned_masked = s * pred_depth_masked + t
        pred_aligned_masked = torch.clip(pred_aligned_masked, min=1e-5, max=None)

        if return_aligned:
            # Apply same s, t to the full frame for visualization
            depth_aligned.append(s * pred_frame + t)

        for met, met_f in metrics.items():
            val = met_f(pred_aligned_masked, gt_depth_masked)
            out_dict.setdefault(met, []).append(val)
    
    # 4. Aggregate
    for met, vals in out_dict.items():
        out_dict[met] = np.mean(vals).tolist()

    if return_aligned:
        return out_dict, torch.stack(depth_aligned, 0)
    else:
        return out_dict

def cam_eval(pred, gt):
    all_rel_rangle_deg, all_rel_tangle_deg = camera_to_rel_deg(pred, gt, "cpu", 1)
    final_Auc_5 = calculate_auc(all_rel_rangle_deg, all_rel_tangle_deg, max_threshold=5).item()
    final_Auc_10 = calculate_auc(all_rel_rangle_deg, all_rel_tangle_deg, max_threshold=10).item()
    final_Auc_15 = calculate_auc(all_rel_rangle_deg, all_rel_tangle_deg, max_threshold=15).item()
    final_Auc_20 = calculate_auc(all_rel_rangle_deg, all_rel_tangle_deg, max_threshold=20).item()
    final_Auc_30 = calculate_auc(all_rel_rangle_deg, all_rel_tangle_deg, max_threshold=30).item()

    final_Racc_5 = (all_rel_rangle_deg < 5).float().mean().item()
    final_Racc_10 = (all_rel_rangle_deg < 10).float().mean().item()
    final_Racc_15 = (all_rel_rangle_deg < 15).float().mean().item()
    final_Racc_20 = (all_rel_rangle_deg < 20).float().mean().item()
    final_Racc_30 = (all_rel_rangle_deg < 30).float().mean().item()

    final_Tacc_5 = (all_rel_tangle_deg < 5).float().mean().item()
    final_Tacc_10 = (all_rel_tangle_deg < 10).float().mean().item()
    final_Tacc_15 = (all_rel_tangle_deg < 15).float().mean().item()
    final_Tacc_20 = (all_rel_tangle_deg < 20).float().mean().item()
    final_Tacc_30 = (all_rel_tangle_deg < 30).float().mean().item()

    return {
        "R@5": final_Racc_5,
        "R@10": final_Racc_10,
        "R@15": final_Racc_15,
        "R@20": final_Racc_20,
        "R@30": final_Racc_30,
        "T@5": final_Tacc_5,
        "T@10": final_Tacc_10,
        "T@15": final_Tacc_15,
        "T@20": final_Tacc_20,
        "T@30": final_Tacc_30,
        "AUC@5": final_Auc_5,
        "AUC@10": final_Auc_10,
        "AUC@15": final_Auc_15,
        "AUC@20": final_Auc_20,
        "AUC@30": final_Auc_30
    }