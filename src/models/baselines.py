import torch
import roma
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from einops import rearrange
from src.utils.gaussians_utils import save_ply, inverse_sigmoid, render, psnr, \
      depth_tv_loss, NormalSupervisionLoss, add_safety_gaussians
from src.utils.geometry_utils import closed_form_inverse_se3, matrix_to_quaternion, quaternion_to_matrix
from src.utils.loss_utils import l1_loss
from src.utils.eval_utils import cam_eval, depth_eval
from src.utils.metrics_utils import compute_psnr, compute_ssim, compute_lpips, evaluate_intrinsics
from src.utils.alignment_utils import get_camera_centers


device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

class DA3GaussianModel(torch.nn.Module):

    def __init__(self,
                 encoder='depthanything3_giant',
                 device='cuda:0'):

        super(DA3GaussianModel, self).__init__()
        
        #gaussian decoder parameters
        self.encoder_name = encoder
        self.device = device

        #training options
        self.training_mode = False
        self.use_depth_loss = True#(self.depth_mode == 'predict')
        self.use_depth_conf = False
        self.use_opacity_loss = True
        self.use_normals_loss = True
        self.use_depth_tv_loss = False
        self.encoder_freezed = False
        self.predict_conf = False
        self.use_lpips = False
        self.init_model()

        self.nvs_scale_align = True
        self.rasterizer = 'gsplat'
            
    def init_model(self):
        
        if 'depthanything3' in self.encoder_name:
            from src.models.geometry_encoders.depthanything3_encoder import DepthAnything3_Encoder
            model_type = self.encoder_name.split('_')[-1]
            if model_type == 'small':
                self.encoder = DepthAnything3_Encoder.from_pretrained("depth-anything/DepthAnythingV3-Small", model_name='da3-small').to(self.device)
            elif model_type == 'base':
                self.encoder = DepthAnything3_Encoder.from_pretrained("depth-anything/DepthAnythingV3-Base", model_name='da3-base').to(self.device)
            elif model_type == 'large':
                self.encoder = DepthAnything3_Encoder.from_pretrained("depth-anything/DepthAnythingV3-Large", model_name='da3-large').to(self.device)
            elif model_type == 'giant':
                self.encoder = DepthAnything3_Encoder.from_pretrained("depth-anything/DepthAnythingV3-Giant", model_name='da3-giant').to(self.device)
            patch_size = 14
        else:
            raise NotImplementedError


    def load_from_checkpoint(self, ckpt_folder):
            
        print("Loading model from weights...")
        
        encoder_ckpt = os.path.join(ckpt_folder,'3D_foundation_model.pth')
        if os.path.exists(encoder_ckpt):
            self.encoder.load_state_dict(torch.load(encoder_ckpt, weights_only=False))
        else:
            print('NO 3D ENCODER CHECKPOINTS, SKIP LOADING FOR THIS MODULE')
        print("Loading done.")

    def save(self, ckpt_folder):
        if not self.encoder_freezed:
            encoder_ckpt = os.path.join(ckpt_folder,'3D_foundation_model.pth')
            torch.save(self.encoder.state_dict(), encoder_ckpt)

    def forward(self, images, target_views=None):
        device = images.device
        
        #Encoder inference (geometry)
       
        geometry_data = self.encoder(images,store_embeddings=True, infer_gs=True)
        
        gaussians = geometry_data["gaussians"]
        geometry_data["gaussians"] = {}
        
    
        
        if self.training_mode:
            raise NotImplementedError
            return gaussians, geometry_data, loss[None], log_dict
        else:
            return gaussians, geometry_data
        
    def confidence_fusion(self, gaussians, batch_idx=0):
        #Do nothing, here for evaluation compatibility
        pruned_gaussians = {}

        for att in gaussians:
            pruned_gaussians[att] = gaussians[att][batch_idx]
        
        return pruned_gaussians
    
    def render_image(self, gaussians_attributes, extrinsic, intrinsic, H, W, sources_views='all', conf_fusion=False,
                     add_safety_gaussian=False, target_view=None, batch_idx=0,render_depth_normal=True,
                     confidence_visualization=False, bg_color=torch.tensor([0.,0.,0.]).cuda()):
        sources_gaussians = {}
        
        for att in gaussians_attributes:
            if att != 'depths':
                if sources_views == 'all':
                    sources_gaussians[att] = gaussians_attributes[att][batch_idx].float()
                else:
                    sources_gaussians[att] = gaussians_attributes[att][batch_idx][sources_views].float()
                #if len(sources_views)>1:
                if 'features' in att:
                    sources_gaussians[att] = rearrange(sources_gaussians[att], 'V N S D -> (V N) S D').float()
                else:
                    sources_gaussians[att] = rearrange(sources_gaussians[att], 'V N D -> (V N) D').float()
            
        if add_safety_gaussian:
            with torch.no_grad():
                sources_gaussians = add_safety_gaussians(sources_gaussians, extrinsic.detach(), intrinsic.detach(), batch_idx)

        
        rendered_image, blend_alphas, blend_semantics, blend_depths, blend_normals, extra_data = render(sources_gaussians,extrinsic,intrinsic,H,W,
                                                                                                        conf_fusion=conf_fusion,render_depth_normal=render_depth_normal,
                                                                                                        confidence_visualization=confidence_visualization,
                                                                                                        bg_color=bg_color, rasterizer=self.rasterizer)
        return rendered_image, blend_depths, blend_normals, blend_alphas
    
    def save_gaussians(self, path, gaussians_attributes, batch_idx=0, conf_fusion=True,save_each_frame=False, conf_fusion_thresh=None, return_gaussians=False):
        batch_size = gaussians_attributes['opacities'].shape[0]
        num_images = gaussians_attributes['opacities'].shape[1]
        if return_gaussians:
            ret_gaussians = []
        
        #for b in range(batch_size):
        b = batch_idx
        saved_gaussians = {}
        if save_each_frame:
            for k in range(num_images):
                for att in gaussians_attributes:
                    if att != 'depths':
                        saved_gaussians[att] = gaussians_attributes[att][b,k]
                save_ply(path+f'_scene_{b}_{k}.ply',saved_gaussians)

        for att in gaussians_attributes:
            if att != 'depths':
                if 'features' in att:
                    saved_gaussians[att] = rearrange(gaussians_attributes[att][b], 'V N S D -> (V N) S D')
                else:
                    saved_gaussians[att] = rearrange(gaussians_attributes[att][b], 'V N D -> (V N) D')
        
        save_ply(path,saved_gaussians)
        
        if return_gaussians:
            return ret_gaussians
    
    def training_setup(self, log_dir, num_gpus):
        
        self.log_dir = log_dir
        self.training_mode = True
        
    def align_target_views(self, source_images, target_images,source_extrinsics=None, scale_align=False):
        
        all_images =  torch.cat([source_images,target_images],dim=1)
        
        with torch.no_grad():
            res = self.encoder(all_images, store_embeddings=False, infer_gs=False)
        
        pred_source = res["extrinsics"][:,:source_images.shape[1]]
        pred_target = res["extrinsics"][:,source_images.shape[1]:]

        if scale_align:
            pred_train_cam_centers = get_camera_centers(pred_source)
            prev_train_cam_centers = get_camera_centers(source_extrinsics)
            scale = torch.mean(prev_train_cam_centers[:,1:],dim=(1,2)) / torch.mean(pred_train_cam_centers[:,1:],dim=(1,2))
            aligned_test_extrinsics = pred_target
            aligned_test_extrinsics[:,1:,:3,3] *= scale[:,None,None]
        else:
            aligned_test_extrinsics = pred_target
            scale = torch.ones(source_images.shape[0]).float().to(aligned_test_extrinsics.device)
            
        
        target_data = {}
        target_data['extrinsics'] = aligned_test_extrinsics
        target_data['depths'] = res["depths"][:,source_images.shape[1]:] * scale[:,None,None,None,None]
        target_data['depth_conf'] = res["depth_conf"][:,source_images.shape[1]:]
        target_data['intrinsics'] = res["intrinsics"][:,source_images.shape[1]:]
        return target_data
