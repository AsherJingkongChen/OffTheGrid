import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from offthegrid.utils.loss_utils import check_and_fix_inf_nan
from vggt.vggt.utils.pose_enc import quat_to_mat

# Modify VGGT class to return latent tokens as well as geometry data
class VGGTOmega_Encoder(VGGTOmega):
    def __init__(self, img_size=512, patch_size=16, embed_dim=1024):
        
        super().__init__()
        self.point_head = None
        self.track_head = None

    def forward(self, images: torch.Tensor, store_embeddings=False) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images

        if store_embeddings:
            predictions["embeddings"] = aggregated_tokens_list

            
        #return predictions
        res = {}
            
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        res["extrinsics"] = extrinsic
        res["intrinsics"] = intrinsic
        res["cam_params"] = predictions["pose_enc"]
        res["depths"] = predictions["depth"].permute(0,1,4,2,3)
        res["depth_conf"] = torch.log(predictions["depth_conf"][:,:,None])
        if "world_points" in predictions:
            res['pointmaps'] = predictions["world_points"].permute(0,1,4,2,3)
            res["pointmaps_conf"] = torch.log(predictions["world_points_conf"][:,:,None])

        if store_embeddings:
            if len(predictions["embeddings"])>20:
                res["embeddings"] = [predictions["embeddings"][k] for k in [4, 11, 17, 23]]
            else:
                res["embeddings"] = [predictions["embeddings"][k] for k in [4, 7, 9, 11]]
        
        return res



# Modify this function to clamp field of view values to avoid NaN 
def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = check_and_fix_inf_nan(pose_encoding[..., :3],'translation vectors',hard_max=100)
        quat = check_and_fix_inf_nan(pose_encoding[..., 3:7],'quaternions',hard_max=None) 
        fov_h = check_and_fix_inf_nan(pose_encoding[..., 7],'fov_h',hard_max=None, replace_value=1) 
        fov_w = check_and_fix_inf_nan(pose_encoding[..., 8],'fov_w',hard_max=None, replace_value=1) 
        fov_h = torch.clamp(fov_h, min=0.1, max=2.6)
        fov_w = torch.clamp(fov_w, min=0.1, max=2.6)
        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics
