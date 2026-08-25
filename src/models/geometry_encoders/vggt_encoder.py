import torch
#from submodules.vggt.vggt.models.vggt import VGGT
#from submodules.vggt.vggt.utils.pose_enc import quat_to_mat
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import quat_to_mat
from src.utils.loss_utils import check_and_fix_inf_nan

# Modify VGGT class to return latent tokens as well as geometry data

class VGGT_Encoder(VGGT):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        
        super().__init__()
        self.point_head = None
        self.track_head = None
        self.camera_head.empty_pose_tokens.requires_grad_(False)

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, store_embeddings=False):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        with torch.amp.autocast('cuda',enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

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
        
        #for k in res:

        #    if k == 'embeddings':
        #        res[k] = [check_and_fix_inf_nan(res[k][i],k,hard_max=None) for i in range(len(res[k]))]
        #        for i in range(len(res[k])):
        #            print(k,i, torch.min(res[k][i]), torch.max(res[k][i]))
        #    else:
        #        res[k] = check_and_fix_inf_nan(res[k],k,hard_max=None)
        #        print(k, torch.min(res[k]), torch.max(res[k]))
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
