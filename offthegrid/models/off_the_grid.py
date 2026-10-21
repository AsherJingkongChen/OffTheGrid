import torch
import roma
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from einops import rearrange
from offthegrid.utils.gaussians_utils import save_ply, inverse_sigmoid, render, psnr, \
      depth_tv_loss, NormalSupervisionLoss, add_safety_gaussians
from offthegrid.utils.geometry_utils import closed_form_inverse_se3, matrix_to_quaternion, quaternion_to_matrix
from offthegrid.utils.loss_utils import l1_loss
from offthegrid.utils.eval_utils import cam_eval, depth_eval
from offthegrid.utils.metrics_utils import compute_psnr, compute_ssim, compute_lpips, evaluate_intrinsics
from offthegrid.utils.alignment_utils import get_camera_centers

from fused_ssim import fused_ssim

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class OffTheGrid(torch.nn.Module):

    def __init__(self,
                 encoder='vggt',
                 decoder='multires_detector',
                 geometry_mode='depth',
                 num_gaussians_per_patch=16,
                 predict_conf=True,
                 use_features=True,
                 device='cuda:0'):
        super(OffTheGrid, self).__init__()
        
        #gaussian decoder parameters
        self.encoder_name = encoder
        self.decoder_name = decoder
        self.geometry_mode = geometry_mode
        self.num_gaussians_per_patch = num_gaussians_per_patch
        self.predict_conf = predict_conf
        self.use_features = use_features
        self.device = device
        

        #gaussians parameters
        self.rasterizer = 'gsplat'
        self.color_mode = 'prior'
        self.depth_mode = 'prior'
        self.conf_mode = 'after_backbone'#'gaussians'#'conf_map'
        self.gaussians_shape = 'disk'

        #training options
        self.training_mode = False
        self.use_depth_loss = True#(self.depth_mode == 'predict')
        self.use_depth_conf = False
        self.use_opacity_loss = True
        self.use_normals_loss = True
        self.use_depth_tv_loss = False
        self.encoder_freezed = False

        self.use_lpips = True
        self.init_model()

        self.nvs_scale_align = True
        if self.encoder_name == 'vggt':
            self.nvs_scale_align = False

        self.train_density_mode = 'adaptive_entropy_image'
        self.test_density_mode = 'adaptive_entropy_image'
        
            
    def init_model(self):
        
        if self.encoder_name == 'vggt':
            from offthegrid.models.geometry_encoders.vggt_encoder import VGGT_Encoder

            self.encoder = VGGT_Encoder().to(self.device)
            if self.geometry_mode == 'depth':
                self.encoder.point_head = None
            else:
                self.encoder.depth_head = None
            patch_size = 14
        elif self.encoder_name == 'mapanything':
            raise NotImplementedError
        elif 'depthanything3' in self.encoder_name:
            from offthegrid.models.geometry_encoders.depthanything3_encoder import DepthAnything3_Encoder
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
        elif self.encoder_name == 'vggt-omega':
            from offthegrid.models.geometry_encoders.vggt_omega_encoder import VGGTOmega_Encoder
            self.encoder = VGGTOmega_Encoder().to(self.device)
            patch_size = 16


        if self.decoder_name == 'offthegrid':
            from offthegrid.models.gaussians_decoders.off_the_grid_decoder import OffTheGridDecoder
            self.gaussians_decoder = OffTheGridDecoder(parent_model=self.encoder_name,
                                    geometry_mode=self.geometry_mode,
                                    num_scales=len(self.num_gaussians_per_patch),
                                    num_gaussians_per_patch=self.num_gaussians_per_patch,
                                    patch_size=patch_size,
                                    depth_input=True,
                                    depth_mode=self.depth_mode,
                                    features_mode='depatchify' if self.use_features else None,
                                    conf_mode = self.conf_mode,
                                    color_mode=self.color_mode,
                                    gaussians_shape=self.gaussians_shape).to(self.device)

                
        if self.decoder_name == 'offthegrid_pretrain':
            from offthegrid.models.gaussians_decoders.off_the_grid_decoder import OffTheGridDecoder
            self.gaussians_decoder = OffTheGridDecoder(parent_model=self.encoder_name,
                                    geometry_mode=self.geometry_mode,
                                    num_scales=len(self.num_gaussians_per_patch),
                                    num_gaussians_per_patch=self.num_gaussians_per_patch,
                                    patch_size=patch_size,
                                    depth_input=True,
                                    depth_mode=self.depth_mode,
                                    features_mode='zeros',
                                    conf_mode = self.conf_mode,
                                    color_mode=self.color_mode,
                                    gaussians_shape=self.gaussians_shape).to(self.device)
        if self.decoder_name == 'multires_detector':
            from offthegrid.models.gaussians_decoders.multires_detector import MultiResGaussiansDecoder
            
            self.gaussians_decoder = MultiResGaussiansDecoder(parent_model=self.encoder_name,
                                                geometry_mode=self.geometry_mode,
                                                num_gaussians_per_patch=self.num_gaussians_per_patch,#[32],#[8,16,32],#[16,32,64],#
                                                patch_size=patch_size,
                                                depth_input=True,
                                                depth_mode=self.depth_mode,
                                                features_mode='depatchify' if self.use_features else None,
                                                color_mode=self.color_mode,
                                                gaussians_shape=self.gaussians_shape).to(self.device)
        if self.decoder_name == 'pixel_aligned':
            from offthegrid.models.gaussians_decoders.pixel_aligned import PixelAlignedDecoder
            self.gaussians_decoder = PixelAlignedDecoder(parent_model=self.encoder_name,
                                                geometry_mode=self.geometry_mode,
                                                decoder_model='MLP',
                                                num_gaussians_per_patch=self.num_gaussians_per_patch,
                                                patch_size=patch_size,
                                                depth_input=True,
                                                depth_mode=self.depth_mode,
                                                features_mode='depatchify' if self.use_features else None,
                                                color_mode=self.color_mode,
                                                gaussians_shape='balls',
                                                add_offset=False).to(self.device)
        

    def load_from_checkpoint(self, ckpt_folder):
            
        print("Loading model from weights...")
        
        encoder_ckpt = os.path.join(ckpt_folder,'3D_foundation_model.pth')
        gaussian_head_ckpt = os.path.join(ckpt_folder,'gaussian_decoder_weights.pth')
        if os.path.exists(encoder_ckpt):
            self.encoder.load_state_dict(torch.load(encoder_ckpt, weights_only=False))
        else:
            print('NO 3D ENCODER CHECKPOINTS, SKIP LOADING FOR THIS MODULE')
        if os.path.exists(gaussian_head_ckpt):
            self.gaussians_decoder.load_state_dict(torch.load(gaussian_head_ckpt, weights_only=False),strict=False)
        else:
            print('NO GAUSSIANS DECODER CHECKPOINTS, SKIP LOADING FOR THIS MODULE')
        print("Loading done.")

    @classmethod
    def from_pretrained(cls, path):
        """Build the released checkpoint in the paper evaluation setting.

        Layout:
            path/3D_foundation_model.pth        # fine-tuned VGGT encoder
            path/gaussian_decoder_weights.pth   # detection decoder
        """
        model = cls(encoder='vggt',
                    decoder='offthegrid',
                    geometry_mode='depth',
                    num_gaussians_per_patch=[16, 32, 64],
                    predict_conf=True,
                    use_features=True,
                    device='cpu')
        model.encoder.load_state_dict(
            torch.load(os.path.join(path, '3D_foundation_model.pth'), map_location='cpu', mmap=True))
        model.gaussians_decoder.load_state_dict(
            torch.load(os.path.join(path, 'gaussian_decoder_weights.pth'), map_location='cpu', mmap=True))
        return model.eval()

    def save(self, ckpt_folder):
        if not self.encoder_freezed:
            encoder_ckpt = os.path.join(ckpt_folder,'3D_foundation_model.pth')
            torch.save(self.encoder.state_dict(), encoder_ckpt)
        gaussian_head_ckpt = os.path.join(ckpt_folder,'gaussian_decoder_weights.pth')
        torch.save(self.gaussians_decoder.state_dict(), gaussian_head_ckpt)

    def forward(self, images, target_views=None, encoder_only=False):
        device = images.device
        
        #Encoder inference (geometry)
        if self.encoder_freezed:
            with torch.no_grad():
                geometry_data = self.encoder(images,store_embeddings=not encoder_only)
        else:
            geometry_data = self.encoder(images,store_embeddings=not encoder_only)
        
        if encoder_only:
            return geometry_data

        embeddings = geometry_data["embeddings"]
        extrinsics = geometry_data["extrinsics"]
        intrinsics = geometry_data["intrinsics"]
        if self.geometry_mode == 'depth':
            depth_map = geometry_data["depths"]
            depth_conf = geometry_data["depth_conf"]
            pointmap = None
            point_conf = None
        else:
            depth_map = None
            depth_conf = None
            pointmap = geometry_data['pointmaps']
            point_conf = geometry_data["pointmaps_conf"]
        
        

        #Decoder inference (gaussians prediction)
        if 'offthegrid' in self.decoder_name:
            if self.training_mode:
                self.gaussians_decoder.density_mode = self.train_density_mode#'adaptive_entropy_image'
            else:
                self.gaussians_decoder.density_mode = self.test_density_mode#'adaptive_entropy_conf_image'

        with torch.autocast(images.device.type, enabled=False):
            embeddings = [e.float() for e in embeddings]
            gaussians = self.gaussians_decoder(images, embeddings, depth_map, depth_conf, pointmap, point_conf, extrinsics, intrinsics)
        
        
        if self.training_mode:
            #Reconstruction of source views
            loss, log_dict = self.render_and_compute_losses(gaussians,
                                                            extrinsics,
                                                            intrinsics,
                                                            images,
                                                            depth_map,
                                                            use_depth_loss=self.use_depth_loss,
                                                            use_normals_loss=self.use_normals_loss,
                                                            use_opacity_loss=self.use_opacity_loss,
                                                            use_lpips=self.use_lpips,
                                                            self_render=self.predict_conf,
                                                            use_conf_loss=True,
                                                            alpha_masks_in_loss=False,
                                                            tto=False)

            if target_views is not None:
                #Reconstruction of target views
                target_images = target_views['images']
                target_depths = target_views['depths']
                target_extrinsics = target_views['extrinsics']
                target_intrinsics = target_views['intrinsics']
                target_view_loss, target_log_dict = self.render_and_compute_losses(gaussians,
                                                            target_extrinsics,
                                                            target_intrinsics,
                                                            target_images,
                                                            target_depths,
                                                            use_depth_loss=False,
                                                            use_normals_loss=False,
                                                            use_opacity_loss=False,
                                                            use_lpips=False,
                                                            self_render=False,
                                                            use_conf_loss=False,
                                                            alpha_masks_in_loss=True,
                                                            tto=False,
                                                            name="target_views_loss")
                loss += target_view_loss
                log_dict = {**target_log_dict, **log_dict}
            return gaussians, geometry_data, loss[None], log_dict
        else:
            return gaussians, geometry_data
        
    
    
    def render_image(self, gaussians_attributes, extrinsic, intrinsic, H, W, sources_views='all', conf_fusion=False,
                     add_safety_gaussian=False, target_view=None, batch_idx=0,render_depth_normal=True,
                     confidence_visualization=False, bg_color=None):
        if bg_color is None:
            bg_color = extrinsic.new_zeros(3)
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
        
            

    def confidence_fusion(self, gaussians, batch_idx=0):
        pruned_gaussians = {}

        if self.use_features:
            opacities = torch.sigmoid(gaussians['opacities'][batch_idx]) * gaussians['conf'][batch_idx]
        else:
            opacities = torch.sigmoid(gaussians['opacities'][batch_idx])

            
        mask = (opacities > 0.1)[:,:,0]
        mask = torch.ones_like(mask)
        if torch.sum(mask)==0:
            return {}
        else:
            for att in gaussians:
                pruned_gaussians[att] = gaussians[att][batch_idx][mask]
            pruned_gaussians['opacities'] = inverse_sigmoid(opacities[mask],epsilon=1e-5)
        return pruned_gaussians

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
        
        if conf_fusion:
            opacities = torch.sigmoid(saved_gaussians['opacities']) * saved_gaussians['conf']
            
            if conf_fusion_thresh is not None:
                depth_conf = saved_gaussians['depth_conf'] > conf_fusion_thresh
                opacities *= depth_conf
            saved_gaussians['opacities'] = inverse_sigmoid(opacities,epsilon=1e-5)
            mask = (opacities > 0.1)[:,0]
            if torch.sum(mask)==0:
                return 0#continue
            #mask = (saved_gaussians['conf'] > 0.1)[:,0]
            for att in saved_gaussians:
                saved_gaussians[att] = saved_gaussians[att][mask]
        
        save_ply(path,saved_gaussians)
        
        if return_gaussians:
            return ret_gaussians
    
    def training_setup(self, log_dir, num_gpus):
        
        self.log_dir = log_dir
        self.training_mode = True
        if self.use_lpips:
            from lpips import LPIPS
            self.lpips_model = LPIPS(net="vgg", eval_mode=False).to(self.device).eval()
            #self.lpips_model = torch.nn.DataParallel(self.lpips_model, device_ids=[i for i in range(num_gpus)]).to(self.device)
        
        self.normals_loss_module = NormalSupervisionLoss()
    
    def render_and_compute_losses(self,
                                gaussians_attributes,
                                extrinsic,
                                intrinsic,
                                images,
                                depth_map,
                                use_depth_loss=True,
                                use_normals_loss=True,
                                use_opacity_loss=True,
                                use_lpips=False,
                                self_render=True,
                                use_conf_loss=True,
                                tto=False,
                                alpha_masks_in_loss=False,
                                name='rendering_loss'):
        with torch.amp.autocast('cuda',enabled=False):
            # Cast inputs to float32 to prevent underflow/overflow in loss computation
            gaussians_attributes = {k: v.float() for k, v in gaussians_attributes.items()}
            extrinsic = extrinsic.float()
            intrinsic = intrinsic.float()
            images = images.float()
            if depth_map is not None:
                depth_map = depth_map.float()
            
            log_dict = {}
            batch_size, num_images, C, H, W = images.shape
            current_device = images.device
            renders = torch.zeros(batch_size, num_images,1+self_render,3,H,W).to(current_device)
            if use_depth_loss:
                rendered_depths = torch.zeros(batch_size, num_images,1,1,H,W).to(current_device)
            if use_opacity_loss or alpha_masks_in_loss:
                alpha_masks = torch.zeros(batch_size, num_images,1,1,H,W).to(current_device)
            if use_normals_loss:
                rendered_normals = torch.zeros(batch_size, num_images,3,H,W).to(current_device)
                target_normals = torch.zeros(batch_size, num_images,3,H,W).to(current_device)

            for b in range(batch_size):
                for i in range(num_images):
                    random_bg = torch.rand(3).to(current_device)
                    extr = extrinsic[b,i].float()
                    intr = intrinsic[b,i].float()
                    source_views = 'all'#[t for t in range(num_images)]
                    rendered_image, blend_depths, blend_normals, blend_alphas = self.render_image(gaussians_attributes,extr,intr,H,W,
                                                                                                source_views,conf_fusion=self.predict_conf,
                                                                                                add_safety_gaussian=False,target_view=i, batch_idx=b,
                                                                                                render_depth_normal=use_normals_loss or use_depth_loss,
                                                                                                bg_color=random_bg)
                    if rendered_image is not None:
                        renders[b,i,0] = rendered_image
                        if use_depth_loss:
                            rendered_depths[b,i,0] = blend_depths[None]
                        if use_opacity_loss or alpha_masks_in_loss:
                            alpha_masks[b,i,0] = blend_alphas[None]
                        if use_normals_loss:
                            rendered_normals[b,i] = blend_normals

                    if self_render:
                        rendered_image, blend_depths, blend_normals, blend_alphas = self.render_image(gaussians_attributes,extr,intr,H,W,
                                                                                                    [i],conf_fusion=False,
                                                                                                    add_safety_gaussian=False,target_view=i, batch_idx=b,
                                                                                                    render_depth_normal=use_normals_loss or use_depth_loss,
                                                                                                    bg_color=random_bg)
                        if rendered_image is not None:
                            renders[b,i,1] = rendered_image

                    if use_normals_loss:
                        if self.rasterizer == 'gsplat':
                            target_normals[b,i] = self.normals_loss_module.compute_normal_from_depth_world(depth_map[b,i], intr, extr, H, W)
                        else:
                            target_normals[b,i] = self.normals_loss_module.compute_normal_from_depth_cam(depth_map[b,i], intr, H, W)
            loss = 0
            if self_render:
                self_renderings = renders[:,:,1]
                renders = renders[:,:,:1]
                self_rgb_loss = l1_loss(self_renderings, images)
                log_dict[f"{name}/self_rgb_loss"] = self_rgb_loss.detach()[None]
                loss += self_rgb_loss

            if use_lpips:
                lpips_model = self.lpips_model.to(current_device)
                #save memory by applying on a single image per scene
                rand_frame = torch.randint(num_images, size=(1,))[0].item()
                lpipsloss = lpips_model(renders[:,rand_frame,0], images[:,rand_frame]).mean()
                log_dict[f"{name}/lpips_loss"] = lpipsloss.detach()[None]

            images = rearrange(images, 'B V C H W -> (B V) C H W')
            renders = rearrange(renders, 'B V R C H W -> (B V R) C H W')
            if alpha_masks_in_loss:
                alpha_masks = rearrange(alpha_masks, 'B V R C H W -> (B V R) C H W')
                rgb_loss = l1_loss(renders, images, alpha_masks)
            else:
                rgb_loss = l1_loss(renders, images)
            
            ssim_loss = fused_ssim(renders, images).mean()
            
            log_dict[f"{name}/rgb_loss"] = rgb_loss.detach()[None]
            log_dict[f"{name}/ssim_loss"] = ssim_loss.detach()[None]
            photometric_loss = 0.8 * rgb_loss + 0.2 * ssim_loss
            
            if use_lpips:
                photometric_loss += 0.2 * lpipsloss

            loss += photometric_loss       

            if use_depth_loss:
                rendered_depths = rearrange(rendered_depths, 'B V R C H W -> (B V R) C H W')
                target_depth = rearrange(depth_map, 'B V C H W -> (B V) C H W')
                if False:#use_depth_conf:
                    target_depth_conf = depth_conf.detach()
                    target_depth_conf = rearrange(target_depth_conf, 'B V C H W -> (B V) C H W')
                    depth_loss = l1_loss(rendered_depths, target_depth, target_depth_conf)
                #elif 'nvs_masks' in data:
                #    depth_loss = l1_loss(rendered_depths, target_depth, nvs_masks) * 10
                else:
                    depth_loss = l1_loss(rendered_depths, target_depth)
                log_dict[f"{name}/depth_loss"] = depth_loss.detach()[None]
                loss += 1.0 * depth_loss
            
            if use_normals_loss:
                #target_normals = rearrange(target_normals.expand(-1,-1,1+self_render,-1,-1,-1),'B V R C H W -> (B V R) C H W')
                #rendered_normals = rearrange(rendered_normals,'B V R C H W -> (B V R) C H W')
                rendered_normals = rearrange(rendered_normals,'B V C H W -> (B V) C H W')
                target_normals = rearrange(target_normals,'B V C H W -> (B V) C H W')
                #if 'nvs_masks' in data:
                #    normals_loss = self.normals_loss_module(rendered_normals, target_normals, nvs_masks[:batch_size * num_images])
                #else:
                normals_loss = self.normals_loss_module(rendered_normals, target_normals)
                
                loss += 0.25 * normals_loss
                log_dict[f"{name}/normals_loss"] = normals_loss.detach()[None]
            

            if use_opacity_loss:
                alpha_loss = l1_loss(alpha_masks,torch.ones_like(alpha_masks))
                loss += 1.0 * alpha_loss
                log_dict["regularization_losses/alpha_loss"] = alpha_loss.detach()[None]

            if use_conf_loss:
                if self.predict_conf:
                    opacities = torch.sigmoid(gaussians_attributes['opacities']) * gaussians_attributes['conf']
                else:
                    opacities = torch.sigmoid(gaussians_attributes['opacities'])
                opacity_loss = torch.sin(opacities * torch.tensor(math.pi)).mean()
                loss += 0.005 * opacity_loss
                log_dict["regularization_losses/opacity_loss"] = opacity_loss.detach()[None]

            percentage = torch.sum(torch.sigmoid(gaussians_attributes["opacities"])<0.5) / torch.numel(gaussians_attributes["opacities"])
            log_dict[f"metrics/percentage_transparent"] = percentage[None]
            if self.predict_conf:
                op = torch.sigmoid(gaussians_attributes["opacities"]) * gaussians_attributes["conf"]
                percentage = torch.sum(op<0.5) / torch.numel(gaussians_attributes["opacities"])
                log_dict[f"metrics/percentage_transparent_with fusion"] = percentage[None]

            if tto and False:
                params_to_optimize = ['extrinsics']
                images = rearrange(images, '(B V) C H W -> B V C H W',B=batch_size,V=num_images)
                opt_gaussians, opt_extrinsics = self.test_time_optimization_new(gaussians_attributes, images,  extrinsic, intrinsic, depth_map, num_iterations=20,
                    params_to_optimize=params_to_optimize,shared_intrinsics=False)
                tto_losses = self.compute_tto_losses(opt_gaussians, opt_extrinsics, gaussians_attributes, extrinsic, params_to_optimize)
                for l in tto_losses:
                    loss += tto_losses[l]
                    log_dict[f"tto_losses/{l}"] = tto_losses[l].detach()[None]

            log_dict[f"{name}/total_loss"] = loss.detach()[None]

            
                    
            return loss, log_dict
    
    def compute_regularization_losses(self, gaussians_attributes, log_dict):
        if self.predict_conf:
            opacities = torch.sigmoid(gaussians_attributes['opacities']) * gaussians_attributes['conf']
        else:
            opacities = torch.sigmoid(gaussians_attributes['opacities'])
        opacity_loss = torch.sin(opacities * torch.tensor(math.pi)).mean()
        log_dict["regularization_losses/opacity_loss"] = opacity_loss.detach()[None]

        percentage = torch.sum(torch.sigmoid(gaussians_attributes["opacities"])<0.5) / torch.numel(gaussians_attributes["opacities"])
        log_dict[f"metrics/percentage_transparent"] = percentage[None]
        if self.predict_conf:
            op = torch.sigmoid(gaussians_attributes["opacities"]) * gaussians_attributes["conf"]
            percentage = torch.sum(op<0.5) / torch.numel(gaussians_attributes["opacities"])
            log_dict[f"metrics/percentage_transparent_with fusion"] = percentage[None]
        return 0.01 * opacity_loss

    def validation_step(self, batch, iteration=0, scene_id=0, tb_writer=None, tto=False, render_opacity=True):
        metrics = {}
        batch_size, num_images, C, H, W = batch["images"].shape
        self.training_mode = False

        gaussians, metadata = self.forward(batch["images"])

        H, W = batch["images"].shape[-2:]
        
        if 'target_views' in batch: #First compute target poses
            num_targets = batch["target_views"].shape[1]
            target_data = self.align_target_views(batch["images"], batch["target_views"], metadata['extrinsics'], scale_align=self.nvs_scale_align)

            target_extrinsics = target_data["extrinsics"]
            test_intr_mode = 'mean'
            if test_intr_mode == 'mean':
                target_intrinsics = metadata["intrinsics"].mean(dim=1)[:,None].repeat(1,num_targets,1,1).to(device)
            else:
                target_intrinsics = target_data["intrinsics"]
            target_depths = target_data["depths"]
            target_renders = torch.zeros(batch_size, num_targets,3,H,W).to(device)
            target_rendered_depths = torch.zeros(batch_size, num_targets,1,H,W).to(device)

        

        if num_images > 1 and 'extrinsics' in batch:
            metrics["validation_camera_pose_estimation/AUC@5"] = []
            metrics["validation_camera_pose_estimation/AUC@10"] = []
            metrics["validation_camera_pose_estimation/AUC@15"] = []
            metrics["validation_camera_pose_estimation/AUC@20"] = []
            metrics["validation_camera_pose_estimation/AUC@30"] = []
            for b in range(batch_size):
                #Evaluate camera pose
                pred_poses = metadata["extrinsics"][b]
                pred_poses = closed_form_inverse_se3(pred_poses)
                #pred_poses = data["extrinsics"][0]
                gt_poses = closed_form_inverse_se3(batch["extrinsics"][b])
                cpe = cam_eval(pred_poses, gt_poses)
                for met in metrics:
                    name = met.split('/')[-1]
                    if name in cpe:
                        metrics[met].append(cpe[name])
        
        if 'depths_gt' in batch:
            metrics["validation_depth_estimation/AbsRel"] = []
            metrics["validation_depth_estimation/SqRel"] = []
            metrics["validation_depth_estimation/RMSElinear"] = []
            metrics["validation_depth_estimation/RMSElog"] = []
            metrics["validation_depth_estimation/InvRMSE"] = []
            metrics["validation_depth_estimation/SILogRMSE"] = []
            metrics["validation_depth_estimation/d1"] = []
            metrics["validation_depth_estimation/d2"] = []
            metrics["validation_depth_estimation/d3"] = []
            #Evaluate depth
            for b in range(batch_size):
                pred_depths = metadata["depths"][b].detach().cpu()
                gt_depth = batch["depths_gt"][b].cpu()
                depth_scene_metrics = depth_eval(pred_depths, gt_depth, return_aligned=False, scene_alignment=True)
                for met in metrics:
                    name = met.split('/')[-1]
                    if name in depth_scene_metrics:
                        metrics[met].append(depth_scene_metrics[name])
                    
        if 'intrinsics' in batch:
            #Evaluate intrinsics
            metrics["validation_intrinsics/intrinsics_MAE"] = []
            metrics["validation_intrinsics/intrinsics_angular_error"] = []
            metrics["validation_intrinsics/T@3"] = []
            intr_metrics = evaluate_intrinsics(metadata["intrinsics"], batch["intrinsics"])
            for met in intr_metrics:
                metrics[f"validation_intrinsics/{met}"] += intr_metrics[met]
            #intrinsics_metrics = evaluate_intrinsics(metrics, intrinsic, batch["intrinsics"])

        #Evaluate rendering
        source_renders = torch.zeros(batch_size, num_images,3,H,W).to(device)
        source_depths = torch.zeros(batch_size, num_images,1,H,W).to(device)
        source_normals = torch.zeros(batch_size, num_images,3,H,W).to(device)
        target_normals = torch.zeros(batch_size, num_images,3,H,W).to(device)
        if num_images>1:
            self_renders = torch.zeros(batch_size, num_images,3,H,W).to(device)
            self_depths = torch.zeros(batch_size, num_images,1,H,W).to(device)
            self_normals = torch.zeros(batch_size, num_images,3,H,W).to(device)
        if render_opacity:
            if not self.predict_conf:
                gaussians['conf'] = torch.sigmoid(gaussians['opacities'])
            opacity_renders = torch.zeros(batch_size, num_images,3,H,W).to(device)

        for b in range(batch_size):
            for i in range(num_images):
                extr = metadata['extrinsics'][b,i].float()
                intr = metadata['intrinsics'][b,i].float()
                if self.rasterizer == 'gsplat':
                    target_normals[b,i] = self.normals_loss_module.compute_normal_from_depth_world(metadata['depths'][b,i], intr, extr, H, W)
                else:
                    target_normals[b,i] = self.normals_loss_module.compute_normal_from_depth_cam(metadata['depths'][b,i], intr, H, W)

                #Rendering of source views
                rendered_image, blend_depths, blend_normals, blend_alphas = self.render_image(gaussians,extr,intr,H,W,
                                                                                              'all',conf_fusion=self.predict_conf,
                                                                                              add_safety_gaussian=False,target_view=i, batch_idx=b,
                                                                                              render_depth_normal=True)
                if rendered_image is not None:
                    source_renders[b,i] = rendered_image
                    source_depths[b,i] = blend_depths[None]
                    source_normals[b,i] = blend_normals

                if num_images>1:
                    #Self render (render predicted gaussians in their source view)
                    rendered_image, blend_depths, blend_normals, blend_alphas = self.render_image(gaussians,extr,intr,H,W,
                                                                                                [i],conf_fusion=False,
                                                                                                add_safety_gaussian=False,target_view=i, batch_idx=b,
                                                                                                render_depth_normal=True)
                    if rendered_image is not None:
                        self_renders[b,i] = rendered_image
                        self_depths[b,i] = blend_depths[None]
                        self_normals[b,i] = blend_normals

                if render_opacity and self.conf_mode=='gaussians':

                    conf_render, _, _, _ = self.render_image(gaussians, extr,intr,H,W,
                                                                    sources_views=[i],conf_fusion=False,
                                                                    add_safety_gaussian=False,target_view=i, batch_idx=b,
                                                                    render_depth_normal=False,confidence_visualization=True)
                    if conf_render is not None:
                        opacity_renders[b,i] = conf_render

            if 'target_views' in batch:
                for i in range(num_targets):
                    extr = target_extrinsics[b,i].float()
                    intr = target_intrinsics[b,i].float()
                    rendered_image, blend_depths, blend_normals, blend_alphas = self.render_image(gaussians,extr,intr,H,W,
                                                                                'all',conf_fusion=self.predict_conf,
                                                                                add_safety_gaussian=False,target_view=i, batch_idx=b,
                                                                                render_depth_normal=True)
                    if rendered_image is not None:
                        target_renders[b,i] = rendered_image
                        target_rendered_depths[b,i] = blend_depths[None]
        
            if tb_writer is not None and scene_id+b <5:
                scene_renders = torch.cat([batch["images"][b].detach().cpu(),source_renders[b].detach().cpu()],dim=-2)
                scene_renders = rearrange(scene_renders, 'N C H W -> C H (N W)')
                tb_writer.add_images(f"validation_source_renders/scene_{scene_id+b}", scene_renders[None], global_step=iteration)
                
                scene_depths = torch.cat([metadata['depths'][b].detach().cpu(),source_depths[b].detach().cpu()],dim=-2)
                scene_depths = rearrange(scene_depths, 'N C H W -> C H (N W)').numpy()
                vmin = np.min(scene_depths)
                scene_depths -= vmin
                vmax = np.max(scene_depths)
                scene_depths /= vmax
                depth_visu = plt.cm.magma_r(scene_depths[0])[..., :3]
                depth_visu = torch.from_numpy(depth_visu).permute(2, 0, 1).float()
                tb_writer.add_images(f"validation_source_depths/scene_{scene_id+b}", depth_visu[None], global_step=iteration)
                
                scene_normals = torch.cat([source_normals[b].detach().cpu(),target_normals[b].detach().cpu()],dim=-2)
                scene_normals = rearrange(scene_normals, 'N C H W -> C H (N W)')
                tb_writer.add_images(f"validation_source_normals/scene_{scene_id+b}", scene_normals[None], global_step=iteration)

                if 'target_views' in batch:
                    scene_renders = torch.cat([batch["target_views"][b].detach().cpu(),target_renders[b].detach().cpu()],dim=-2)
                    scene_renders = rearrange(scene_renders, 'N C H W -> C H (N W)')
                    tb_writer.add_images(f"validation_target_renders/scene_{scene_id+b}", scene_renders[None], global_step=iteration)

                    scene_depths = torch.cat([target_depths[b].detach().cpu(),target_rendered_depths[b].detach().cpu()],dim=-2)
                    scene_depths = rearrange(scene_depths, 'N C H W -> C H (N W)').numpy()
                    vmin = np.min(scene_depths)
                    scene_depths -= vmin
                    vmax = np.max(scene_depths)
                    scene_depths /= vmax
                    depth_visu = plt.cm.magma_r(scene_depths[0])[..., :3]
                    depth_visu = torch.from_numpy(depth_visu).permute(2, 0, 1).float()
                    tb_writer.add_images(f"validation_target_depths/scene_{scene_id+b}", depth_visu[None], global_step=iteration)

                if hasattr(self.gaussians_decoder,"image_coordinates"):
                    detected_gaussians = [self.gaussians_decoder.visualize_gaussians_detections(batch["images"], batch_id=b, im_id=i) for i in range(num_images)]
                    detected_gaussians = torch.cat(detected_gaussians, dim=-1)
                    tb_writer.add_images(f"validation_gaussian_detection/scene_{scene_id+b}", detected_gaussians[None], global_step=iteration)

                if render_opacity:
                    if self.conf_mode!='gaussians':
                        opacity_viz = self.gaussians_decoder.conf_map[b].detach().cpu()
                        opacity_viz = rearrange(opacity_viz, 'N C H W -> C H (N W)')
                    else:
                        opacity_viz = rearrange(opacity_renders[b].detach().cpu(), 'N C H W -> C H (N W)')
                    tb_writer.add_images(f"validation_conf_visualization/scene_{scene_id+b}", opacity_viz[None], global_step=iteration)
                
                if 'nvs_masks' in batch:
                    tb_writer.add_images(f"validation_nvs_masks/scene_{scene_id+b}", batch['nvs_masks'][b,i].detach().cpu()[None], global_step=iteration)
        
        
        metrics["validation_source_render/source_PSNR"] = []
        metrics["validation_source_render/source_SSIM"] = []
        metrics["validation_source_render/source_LPIPS"] = []
        if num_images>1:
            metrics["validation_self_render/self_render_PSNR"] = []
            metrics["validation_self_render/self_render_SSIM"] = []
            metrics["validation_self_render/self_render_LPIPS"] = []
        if 'target_views' in batch:
            metrics["validation_target_render/target_PSNR"] = []
            metrics["validation_target_render/target_SSIM"] = []
            metrics["validation_target_render/target_LPIPS"] = []

        for b in range(batch_size):
            metrics["validation_source_render/source_PSNR"] += compute_psnr(source_renders[b].cpu(),batch["images"][b].cpu()).double().tolist()
            metrics["validation_source_render/source_SSIM"] += compute_ssim(source_renders[b].cpu(),batch["images"][b].cpu()).double().tolist()
            metrics["validation_source_render/source_LPIPS"] += compute_lpips(source_renders[b].cpu(),batch["images"][b].cpu()).double().tolist()
            if num_images>1:
                metrics["validation_self_render/self_render_PSNR"] += compute_psnr(self_renders[b].cpu(),batch["images"][b].cpu()).double().tolist()
                metrics["validation_self_render/self_render_SSIM"] += compute_ssim(self_renders[b].cpu(),batch["images"][b].cpu()).double().tolist()
                metrics["validation_self_render/self_render_LPIPS"] += compute_lpips(self_renders[b].cpu(),batch["images"][b].cpu()).double().tolist()
            if 'target_views' in batch:
                metrics["validation_target_render/target_PSNR"] += compute_psnr(target_renders[b].cpu(),batch["target_views"][b].cpu()).double().tolist()
                metrics["validation_target_render/target_SSIM"] += compute_ssim(target_renders[b].cpu(),batch["target_views"][b].cpu()).double().tolist()
                metrics["validation_target_render/target_LPIPS"] += compute_lpips(target_renders[b].cpu(),batch["target_views"][b].cpu()).double().tolist()
        
        for b in range(batch_size):
            self.save_gaussians(f'{self.log_dir}/val_gaussians_scene_{scene_id+b}.ply', gaussians, batch_idx=b, conf_fusion=self.predict_conf)

        self.training_mode = True
        return metrics

    def align_target_views(self, source_images, target_images, source_extrinsics=None, scale_align=False, chunk_size=50):
        B = source_images.shape[0]
        N_source = source_images.shape[1]
        N_target = target_images.shape[1]
        
        # Calculate how many target views we can fit in one forward pass
        max_target_per_chunk = chunk_size - N_source
        
        # Fallback in case source_images alone is larger than or equal to chunk_size
        if max_target_per_chunk <= 0:
            max_target_per_chunk = 1
            
        all_aligned_extrinsics = []
        all_depths = []
        all_depth_conf = []
        all_intrinsics = []
        
        # Process target images in chunks
        for i in range(0, N_target, max_target_per_chunk):
            target_chunk = target_images[:, i:i+max_target_per_chunk]
            all_images_chunk = torch.cat([source_images, target_chunk], dim=1)
            
            with torch.no_grad():
                res = self.encoder(all_images_chunk, store_embeddings=False)
            
            pred_source = res["extrinsics"][:, :N_source]
            pred_target = res["extrinsics"][:, N_source:]

            if scale_align:
                pred_train_cam_centers = get_camera_centers(pred_source)
                prev_train_cam_centers = get_camera_centers(source_extrinsics)
                
                # Calculate scale per-chunk to account for slight drifts in encoder predictions
                scale = torch.mean(prev_train_cam_centers[:, 1:], dim=(1,2)) / torch.mean(pred_train_cam_centers[:, 1:], dim=(1,2))
                
                aligned_test_extrinsics = pred_target.clone()
                # Apply scale to all target views in the current chunk
                aligned_test_extrinsics[:, :, :3, 3] *= scale[:, None, None]
            else:
                aligned_test_extrinsics = pred_target.clone()
                scale = torch.ones(B).float().to(aligned_test_extrinsics.device)
                
            # Store chunk results
            all_aligned_extrinsics.append(aligned_test_extrinsics)
            all_depths.append(res["depths"][:, N_source:] * scale[:, None, None, None, None])
            all_depth_conf.append(res["depth_conf"][:, N_source:])
            all_intrinsics.append(res["intrinsics"][:, N_source:])
            
        # Aggregate chunked results back into the final dictionary
        target_data = {}
        target_data['extrinsics'] = torch.cat(all_aligned_extrinsics, dim=1)
        target_data['depths'] = torch.cat(all_depths, dim=1)
        target_data['depth_conf'] = torch.cat(all_depth_conf, dim=1)
        target_data['intrinsics'] = torch.cat(all_intrinsics, dim=1)
        
        return target_data

    def align_target_views_prev(self, source_images, target_images,source_extrinsics=None, scale_align=False):
        
        all_images =  torch.cat([source_images,target_images],dim=1)
        
        with torch.no_grad():
            res = self.encoder(all_images, store_embeddings=False)
        
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

    def test_time_optimization(self, gaussians, images, depths, extrinsics, intrinsics, num_iterations=100, params_to_optimize=["extrinsics"],shared_intrinsics=True, max_images=-1):

        B, num_images, _, H, W = images.shape        
        for att in gaussians:
            gaussians[att] = gaussians[att].detach()
        images = images.detach()
        depths = depths.detach()
        extrinsics = extrinsics.detach().clone()
        intrinsics = intrinsics.detach()

        l = []
        if "conf" in params_to_optimize:
            _conf = torch.nn.Parameter(gaussians["conf"].detach().float().clone().requires_grad_(True))
            #_conf = torch.nn.Parameter(inverse_sigmoid(gaussians["conf"].float()).requires_grad_(True))
            #_conf = torch.nn.Parameter(torch.zeros_like(gaussians["conf"]).requires_grad_(True))
            l.append({'params': [_conf], 'lr': 0.01, "name": "conf"})
        if "extrinsics" in params_to_optimize:
            with torch.no_grad():
                #c2w_orientation = torch.zeros(B,num_images,4).to(extrinsics.device)
                #c2w_translation = torch.zeros(B,num_images,3).to(extrinsics.device)
                cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics).detach()
                c2w_orientation = matrix_to_quaternion(cam_to_world_extrinsic[...,:3,:3])
                c2w_translation = cam_to_world_extrinsic[...,:3,3]

            c2w_orientation = torch.nn.Parameter(c2w_orientation.float().clone().requires_grad_(True))
            c2w_translation = torch.nn.Parameter(c2w_translation.float().clone().requires_grad_(True))
            l.append({'params': [c2w_translation], 'lr': 0.0001, "name": "c2w_translation"})
            l.append({'params': [c2w_orientation], 'lr': 0.0001, "name": "c2w_orientation"})
        if "intrinsics" in params_to_optimize:
            if shared_intrinsics:
                fxs = intrinsics[:,:1,0,0].detach()
                fys = intrinsics[:,:1,1,1].detach()
            else:
                fxs = intrinsics[:,:,0,0].detach()
                fys = intrinsics[:,:,1,1].detach()
            focals = torch.stack([fxs,fys],dim=-1)
            focals = torch.nn.Parameter(focals.float().clone().requires_grad_(True))
            l.append({'params': [focals], 'lr': 0.1, "name": "focals"})
        if "depth" in params_to_optimize:
            pred_depths = torch.nn.Parameter(gaussians["depths"].detach().float().clone().requires_grad_(True))
            l.append({'params': [pred_depths], 'lr': 0.01, "name": "depth"})
        if "scales" in params_to_optimize:
            _scales = torch.nn.Parameter(torch.exp(gaussians["scales"]).detach().float().clone().requires_grad_(True))
            l.append({'params': [_scales], 'lr': 0.001, "name": "scales"})
        local_orient = self.local_orient.clone().detach()
        if "orientations" in params_to_optimize:
            l.append({'params': [local_orient], 'lr': 0.05, "name": "orientations"})

        optimizer = torch.optim.Adam(l, eps=1e-15)

        if self.use_normals_loss and not hasattr(self,"normals_loss_module"):
            self.normals_loss_module = NormalSupervisionLoss()
        
        for k in range(num_iterations):
            
            if "extrinsics" in params_to_optimize or "intrinsics" in params_to_optimize:
                gaussians['means'] = torch.zeros_like(gaussians['means'])
                gaussians['quats'] = torch.zeros_like(gaussians['quats'])
                
            if "conf" in params_to_optimize:
                gaussians["conf"] = torch.clamp(_conf,0,1)

            local_pos = self.local_pos.clone().detach()

            if "depth" in params_to_optimize:
                local_pos[...,:1] = pred_depths
            
            if "intrinsics" in params_to_optimize:
                fxs = focals[...,0]
                fys = focals[...,1]
                intr = torch.eye(3)[None,None].expand(B,num_images,3,3).cuda()
                intr[...,0,2] = H // 2
                intr[...,1,2] = W // 2
                intr[...,0,0] = fxs
                intr[...,1,1] = fys
            else:
                intr = intrinsics.detach()#[b,:,0]

            if 'scales' in params_to_optimize:
                gaussians["scales"] = torch.log(_scales)
            
            
            if "extrinsics" in params_to_optimize or "intrinsics" in params_to_optimize:
                R_c2w = quaternion_to_matrix(c2w_orientation)
                t_c2w = c2w_translation
                #extrinsics = closed_form_inverse_se3(torch.cat([R_c2w,t_c2w[...,None]],dim=-1).detach()).contiguous() #Need w2c params for rasterization
                extrinsics = torch.cat([R_c2w,t_c2w[...,None]],dim=-1).detach()
                pos, orient, _ = self.gaussians_cam2world(local_pos,local_orient, t_c2w, R_c2w, intr)
                gaussians['means'] = pos
                gaussians['quats'] = orient
            else:
                R_c2w = extrinsics[b,:,:3,:3]
                t_c2w = extrinsics[b,:,:3,3]


                

            loss, log_dict = self.render_and_compute_losses(gaussians,
                                                            extrinsics,
                                                            intr,
                                                            images,
                                                            depths,
                                                            use_depth_loss=True,
                                                            use_normals_loss=False,
                                                            use_opacity_loss=True,
                                                            use_lpips=False,
                                                            self_render=False,
                                                            use_conf_loss=False,
                                                            tto=False)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if "conf" in params_to_optimize:
            gaussians["conf"] = torch.clamp(_conf,0,1)
        if "depth" in params_to_optimize:
            gaussians["depths"] = pred_depths
        if "extrinsics" in params_to_optimize:
            R_c2w = quaternion_to_matrix(c2w_orientation)
            t_c2w = c2w_translation
            extrinsics = closed_form_inverse_se3(torch.cat([R_c2w,t_c2w[...,None]],dim=-1).detach()).contiguous() #Need w2c params for rasterization
            extrinsics = extrinsics[...,:3,:]
        if shared_intrinsics:
            intr = intr.expand(-1,num_images,-1,-1)
        return gaussians, extrinsics.float(), intr.float()

    def compute_tto_losses(self, opt_gaussians, opt_extrinsics, gaussians_attributes, extrinsic, params_to_optimize=[]):
        tto_losses = {}
        
        if "gaussians" in params_to_optimize:
            position_loss = l1_loss(gaussians_attributes["means"], opt_gaussians["means"].detach())
            scale_loss = l1_loss(gaussians_attributes["scales"], opt_gaussians["scales"].detach())
            quat_loss = l1_loss(gaussians_attributes["quats"], opt_gaussians["quats"].detach())
            if self.predict_conf:
                op = torch.sigmoid(gaussians_attributes["opacities"]) * gaussians_attributes["conf"]
            else:
                op = torch.sigmoid(gaussians_attributes["opacities"])
            op_opt = torch.sigmoid(opt_gaussians["opacities"])
            opacity_loss = l1_loss(op, op_opt)
            
            tto_losses['position_loss'] = position_loss
            tto_losses['scale_loss'] = scale_loss
            tto_losses['orientation_loss'] = quat_loss
            tto_losses['opacity_loss'] = opacity_loss
        if "extrinsics" in params_to_optimize:
            transl_loss = l1_loss(extrinsic[...,:3,3], opt_extrinsics[...,:3,3].detach())
            orientation_loss = roma.rotmat_geodesic_distance(extrinsic[...,:3,:3], opt_extrinsics[...,:3,:3].detach()).mean()
            tto_losses['extr_transl_loss'] = transl_loss
            tto_losses['extr_rot_loss'] = orientation_loss
        
        return tto_losses

    def test_time_optimization_new(self, gaussians, images, extrinsics, intrinsics, depths, num_iterations=100, params_to_optimize=["gaussians","extrinsics"],shared_intrinsics=True, max_images=-1):

        B, num_images, _, H, W = images.shape
        for att in gaussians:
            gaussians[att] = gaussians[att].detach()
        images = images.detach()
        extrinsics = extrinsics.detach().clone()
        intrinsics = intrinsics.detach()
        depths = depths.detach()

        use_conf = True
        optimize_color = True

        if not hasattr(self,'normals_loss_module'):
            self.normals_loss_module = NormalSupervisionLoss()

        l = []
        if "extrinsics" in params_to_optimize:
            _orientation = matrix_to_quaternion(extrinsics[:,:,:3,:3])
            _translation = extrinsics[:,:,:3,3]
            _orientation = torch.nn.Parameter(_orientation.requires_grad_(True))
            _translation = torch.nn.Parameter(_translation.requires_grad_(True))
            l.append({'params': [_translation], 'lr': 0.005, "name": "translation"})
            l.append({'params': [_orientation], 'lr': 0.005, "name": "orientation"})

        if "gaussians" in params_to_optimize:
            _means = torch.nn.Parameter(gaussians["means"].float().clone().requires_grad_(True))
            _quats = torch.nn.Parameter(gaussians["quats"].float().clone().requires_grad_(True))
            _scales = torch.nn.Parameter(gaussians["scales"].float().clone().requires_grad_(True))
    
            _opacities = gaussians["opacities"].float().clone().requires_grad_(True)
            if use_conf:
                _conf = gaussians["conf"].float().clone().requires_grad_(True)
                _opacities = inverse_sigmoid(torch.sigmoid(_opacities) * _conf)
            _opacities = torch.nn.Parameter(_opacities)

            l.append({'params': [_means], 'lr': 0.0001, "name": "means"})
            l.append({'params': [_quats], 'lr': 0.01, "name": "quats"})
            l.append({'params': [_scales], 'lr': 0.01, "name": "scales"})
            l.append({'params': [_opacities], 'lr': 0.05, "name": "opacities"})

            if optimize_color:
                _shs = torch.nn.Parameter(gaussians["features_dc"].float().clone().requires_grad_(True))
                l.append({'params': [_shs], 'lr': 0.05, "name": "features_sh"})
        

        optimizer = torch.optim.Adam(l, eps=1e-15)
        
        for k in range(num_iterations):
            if "gaussians" in params_to_optimize:
                optimized_gaussians = {}
                optimized_gaussians['means'] = _means
                optimized_gaussians["quats"] = torch.nn.functional.normalize(_quats,dim=-1)
                optimized_gaussians["scales"] = _scales
                optimized_gaussians["opacities"] = _opacities
                optimized_gaussians["conf"] = torch.ones_like(_opacities)
                if optimize_color:
                    optimized_gaussians["features_dc"] = _shs
                else:
                    optimized_gaussians["features_dc"] = gaussians["features_dc"]
                optimized_gaussians["features_sh"] = gaussians["features_sh"]
            else:
                optimized_gaussians = gaussians
            
            if 'extrinsics' in params_to_optimize:
                quat = torch.nn.functional.normalize(_orientation,dim=-1)
                rot = quaternion_to_matrix(quat)
                extr = torch.cat([rot,_translation[...,None]],dim=-1)
            else:
                extr = extrinsics
            
            intr = intrinsics

            loss, _ = self.render_and_compute_losses(optimized_gaussians,
                                                        extr,
                                                        intr,
                                                        images,
                                                        depths,
                                                        use_depth_loss=True,
                                                        use_normals_loss=False,
                                                        use_opacity_loss=False,
                                                        use_lpips=False,
                                                        self_render=False,
                                                        use_conf_loss=False,
                                                        tto=False)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if "gaussians" in params_to_optimize:
            optimized_gaussians = {}
            optimized_gaussians['means'] = _means
            optimized_gaussians["quats"] = torch.nn.functional.normalize(_quats,dim=-1)
            optimized_gaussians["scales"] = _scales
            optimized_gaussians["opacities"] = _opacities
            optimized_gaussians["conf"] = torch.ones_like(_opacities)
            if optimize_color:
                optimized_gaussians["features_dc"] = _shs
            else:
                optimized_gaussians["features_dc"] = gaussians["features_dc"]
            optimized_gaussians["features_sh"] = gaussians["features_sh"]
        else:
            optimized_gaussians = gaussians
        
        if 'extrinsics' in params_to_optimize:
            quat = torch.nn.functional.normalize(_orientation,dim=-1)
            rot = quaternion_to_matrix(quat)
            extr = torch.cat([rot,_translation[...,None]],dim=-1)
        else:
            extr = extrinsics
        return optimized_gaussians, extr.float()
