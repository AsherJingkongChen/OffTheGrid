import torch
from depth_anything_3.api import DepthAnything3
from einops import rearrange
import torchvision.transforms as T
import math
from src.utils.gaussians_utils import inverse_sigmoid

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class DepthAnything3_Encoder(DepthAnything3):
    def __init__(self, model_name="DA3-LARGE-1.1",**kwargs):
        """
        Initializes the DepthAnything3 encoder.
        """
        #super().__init__(model_name=model_type)
        super().__init__(model_name=model_name, **kwargs)

        self.ref_view_strategy = "first"#"saddle_balanced"
        #self.train()
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        if self.model_name in ["da3-base","da3-small"]:
            self.out_layers = [5, 7, 9, 11]
        elif self.model_name in ['da3-large']:
            self.out_layers = [11, 15, 19, 23]
        elif self.model_name in ["da3-giant"]:
            self.out_layers = [19, 27, 33, 39]

        self.near_depth = 0.02
        self.far_depth = 50

    def forward(self, images: torch.Tensor, store_embeddings=True, use_ray_pose=False, infer_gs=False):
        # Define which layers to extract. 
        export_feat_layers = self.out_layers if store_embeddings else []
        
        ref_view_strategy = self.ref_view_strategy
                
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        #print(autocast_dtype, images.device.type)
        with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
            images = (images - self._resnet_mean.to(images.device).to(images.dtype)) / self._resnet_std.to(images.device).to(images.dtype)
            raw_output = self.model(
                images, 
                None,  # extrinsics
                None,  # intrinsics
                export_feat_layers, 
                infer_gs, 
                use_ray_pose, 
                ref_view_strategy
            )
        
        res = {}
        if "depth" in raw_output:
            res["depths"] = torch.clamp(raw_output["depth"][:,:,None],min=self.near_depth, max=self.far_depth)


        if "depth_conf" in raw_output:
            res["depth_conf"] = torch.log(raw_output["depth_conf"][:,:,None])

        if "extrinsics" in raw_output:
            res["extrinsics"] = raw_output["extrinsics"]
            
        if "intrinsics" in raw_output:
            H, W = images.shape[-2:]
            res["intrinsics"] = torch.zeros_like(raw_output["intrinsics"])
            res["intrinsics"][..., 2, 2] = 1
            res["intrinsics"][..., 0, 2] = W / 2
            res["intrinsics"][..., 1, 2] = H / 2

            f_min_h = (H / 2.0) / math.tan(2.6 / 2.0)
            f_min_w = (W / 2.0) / math.tan(2.6 / 2.0)
            f_max_h = (H / 2.0) / math.tan(0.1 / 2.0)
            f_max_w = (W / 2.0) / math.tan(0.1 / 2.0)

            fx = torch.clamp(raw_output["intrinsics"][..., 0, 0], min=f_min_w, max=f_max_w)
            fy = torch.clamp(raw_output["intrinsics"][..., 1, 1], min=f_min_h, max=f_max_h)

            res["intrinsics"][..., 0, 0] = fx
            res["intrinsics"][..., 1, 1] = fy
            #res["intrinsics"] = raw_output["intrinsics"]

        if store_embeddings and "aux" in raw_output:
            features_dict = dict(raw_output["aux"])
            res["embeddings"] = [rearrange(features_dict[t],'B V h w D -> B V (h w) D') for t in features_dict.keys()]

        if infer_gs:
            gs_obj = raw_output["gaussians"]
            means = gs_obj.means
            scales = torch.log(gs_obj.scales)
            #quats = gs_obj.rotations[..., [3, 0, 1, 2]].contiguous()
            quats = gs_obj.rotations
            feats_dc = gs_obj.harmonics[:, :, :, 0].unsqueeze(2).contiguous()
            feats_sh = gs_obj.harmonics.transpose(-1, -2)[:,:,1:].contiguous()
            opacities = inverse_sigmoid(gs_obj.opacities)[...,None].contiguous()
            res["gaussians"] = {
                "means": rearrange(means,'B (V N) D -> B V N D', V=images.shape[1]),         
                "scales": rearrange(scales,'B (V N) D -> B V N D', V=images.shape[1]),       
                "quats": rearrange(quats,'B (V N) D -> B V N D', V=images.shape[1]),
                "features_dc": rearrange(feats_dc,'B (V N) F D -> B V N F D', V=images.shape[1]),
                "features_sh": rearrange(feats_sh,'B (V N) F D -> B V N F D', V=images.shape[1]),
                "opacities": rearrange(opacities,'B (V N) D -> B V N D', V=images.shape[1]),
            }
            

        return res