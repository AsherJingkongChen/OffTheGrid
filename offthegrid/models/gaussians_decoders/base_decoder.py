import torch
import torchvision
import numpy as np
from einops import rearrange
from torchvision import transforms
from offthegrid.utils.gaussians_utils import patchify, build_rotation, extract_rotation
from offthegrid.vggt.heads.dpt_head import DPTHead
from offthegrid.utils.geometry_utils import closed_form_inverse_se3, pred_to_cam_coords
from offthegrid.models.gaussians_decoders.layers import UNet, MLP

class GaussiansDecoder(torch.nn.Module):
    def __init__(self,
                parent_model='vggt',
                geometry_mode='depth',
                patch_size=14,
                depth_input=True,
                depth_mode='prior',
                features_mode='depatchify',
                conf_mode='gaussians',
                color_mode='prior',
                gaussians_shape='disk'):
        
        super().__init__()

        self.parent_model = parent_model
        self.geometry_mode = geometry_mode
        self.patch_size = patch_size
        self.depth_input = depth_input
        self.depth_mode = depth_mode
        self.features_mode = features_mode
        self.predict_color = (color_mode=='predict')
        self.gaussians_shape = gaussians_shape
        self.conf_mode = conf_mode

        self.visualization_mode = False
        self.save_local_positions = False
        
        self.offset_3d = False
        self.backbone='unet'
        
        self.num_features_channels = 8
        self.image_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) 
        self.d = 32
        
        #Gaussians parameters options
        self.attributes_names = ['means', 'quats', 'scales', 'opacities', 'features_dc', 'features_sh']
        self.predict_conf = self.conf_mode is not None
        self.near_depth = 0.02
        self.far_depth = 50
        self.multiply_scale_by_depth = True
        self.min_scale = 0.0001
        self.max_scale = 0.008
        if self.depth_mode == 'predict':
            self.depth_activation = 'softplus'
        else:
            self.depth_activation = 'identity'

        if self.features_mode is not None:
            #self.features_norm = torch.nn.LayerNorm(self.num_features_channels)
            if self.parent_model=='vggt':
                feat_dim = 2048
            elif 'depthanything3' in self.parent_model:
                model_type = self.parent_model.split('_')[-1]
                if model_type == 'small':
                    feat_dim = 768
                elif model_type == 'base':
                    feat_dim = 768
                elif model_type == 'large':
                    feat_dim = 1024
                elif model_type == 'giant':
                    feat_dim = 1536

            if self.features_mode == 'DPT':
                self.features_head = DPTHead(feat_dim, patch_size, output_dim=self.num_features_channels+1, activation='linear', conf_activation='sigmoid', intermediate_layer_idx=[0,1,2,3])
            elif self.features_mode == 'depatchify':
                if 'vggt' in self.parent_model:
                    self.norm = torch.nn.LayerNorm(2048)
                    self.depatchifier = MLP(input_dim=2048*4, output_dim=patch_size*patch_size*self.num_features_channels, net_depth=2, net_width=1024)
                elif 'depthanything3' in self.parent_model:
                    model_type = self.parent_model.split('_')[-1]
                    if model_type == 'small':
                        feat_dim = 768
                    elif model_type == 'base':
                        feat_dim = 768
                    elif model_type == 'large':
                        feat_dim = 1024
                    elif model_type == 'giant':
                        feat_dim = 1536
                    self.norm = torch.nn.LayerNorm(feat_dim)
                    self.depatchifier = MLP(input_dim=feat_dim*4, output_dim=patch_size*patch_size*self.num_features_channels, net_depth=2, net_width=feat_dim)            
        
        

        in_channels = 3 + (2 + 2*(self.geometry_mode=='pointmap')) * self.depth_input + self.num_features_channels * (self.features_mode is not None)
        if self.backbone == 'unet':
            self.backbone = UNet(n_channels=in_channels,n_classes=self.d)
        else:
            raise NotImplementedError

        if self.conf_mode == 'after_backbone':
            self.conf_head = torch.nn.Sequential(
                torch.nn.Conv2d(self.d, self.d, kernel_size=3, stride=1,padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(self.d, 1 , kernel_size=1, stride=1)
                )

    def features_to_gaussians(self, x, images, depth, depth_conf, pointmap, point_conf, conf_map=None):
        raise NotImplementedError

    def set_coords_grids(self, H, W):
        y_coords = torch.linspace(0, H - 1, H) + 0.5
        x_coords = torch.linspace(0, W - 1, W) + 0.5
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords_map = torch.stack([xx,yy],dim=0)[None]
        coords_map = patchify(coords_map,(self.patch_size,self.patch_size))
        self.coords_grids = coords_map #B N 2 h w
        
    def visualize_gaussians_detections(self, images, batch_id=0, im_id=0):
        coords = torch.floor(self.image_coordinates[batch_id*self.num_images + im_id]).detach().cpu().int()
        image = torchvision.utils.draw_keypoints(torch.zeros_like(images[batch_id,im_id]).cpu(), coords[None], connectivity=None, colors="green", radius=1)
        return image
    
    def compute_gaussians_attributes(self, gaussians, extrinsics, intrinsics):
        
        attribute_dict = {}
        batch_size = gaussians.shape[0]
        num_images = gaussians.shape[1]
        
        #Gaussians position
        if self.save_local_positions:
            self.local_pos = gaussians[...,:3].detach()
            self.local_orient = gaussians[...,3:7].detach()

        #Reshape in a single batch dimension
        gaussians = rearrange(gaussians,'B V N D -> (B V) N D').float()
        extrinsics = rearrange(extrinsics,'B V D1 D2 -> (B V) D1 D2')
        intrinsics = rearrange(intrinsics,'B V D1 D2 -> (B V) D1 D2')

        cam_to_world_extrinsic = closed_form_inverse_se3(extrinsics)
        R_c2w = cam_to_world_extrinsic[:,:3, :3].float()
        t_c2w = cam_to_world_extrinsic[:, :3, 3].float()

        if self.geometry_mode == 'depth':
            if self.offset_3d:
                offset_3d = gaussians[...,-3:] * 0.1
                attribute_dict['offsets_3D'] = offset_3d
            else:
                offset_3d = None
            pos, orient, depths = self.gaussians_cam2world(gaussians[...,:3],gaussians[...,3:7], t_c2w, R_c2w, intrinsics, offset_3d)
            attribute_dict['means'] = pos
            attribute_dict['quats'] = torch.nn.functional.normalize(orient, p=2, dim=-1)
            attribute_dict['depths'] = depths
        else:
            attribute_dict['means'] = gaussians[...,:3]
            attribute_dict['quats'] = gaussians[...,3:7]
        
    
        scale_interp = torch.sigmoid(gaussians[...,7:10])
        scale_interp = (1 - scale_interp) * self.min_scale + scale_interp * self.max_scale
        if self.gaussians_shape == 'disk':
            scale_interp[...,2] = 0.1 * scale_interp[...,2]
        if self.multiply_scale_by_depth and self.geometry_mode == 'depth':
            scale_interp *= depths.view(batch_size * num_images,-1,1).detach()
        
        attribute_dict['scales'] =  torch.log(scale_interp)
        #attribute_dict['opacities'] = inverse_sigmoid(torch.ones_like(gaussians[...,10:11]),epsilon=1e-5)
        attribute_dict['opacities'] = gaussians[...,10:11]
        attribute_dict['features_dc'] = gaussians[...,11:14]
                
        if self.predict_conf:
            if self.conf_mode == 'gaussians':
                attribute_dict['conf'] = torch.sigmoid(gaussians[...,14:15])
            else:
                attribute_dict['conf'] = gaussians[...,14:15]

        for att in attribute_dict:
            attribute_dict[att] = rearrange(attribute_dict[att],'(B V) N D -> B V N D', B=batch_size)
            attribute_dict[att] = attribute_dict[att].contiguous()#.float()

        attribute_dict['features_sh'] = torch.zeros(batch_size,num_images,gaussians.shape[1],0,3).to(gaussians.device)
        attribute_dict['features_dc'] = attribute_dict['features_dc'][...,None,:]

        return attribute_dict
    
    def gaussians_cam2world(self, positions, pred_orientations_cam, c2w_t, c2w_R, intrinsics, offset_3d=None):
        
        if len(pred_orientations_cam.shape) == 4:
            B, V, H, W = pred_orientations_cam.shape
            positions = rearrange(positions, 'B V N D -> (B V) N D')
            pred_orientations_cam = rearrange(pred_orientations_cam, 'B V N D -> (B V) N D')
            c2w_t = rearrange(c2w_t, 'B V D -> (B V) D')
            c2w_R = rearrange(c2w_R, 'B V N D -> (B V) N D')
            intrinsics = rearrange(intrinsics, 'B V N D -> (B V) N D')
            need_reshape = True
        else:
            need_reshape = False

        num_images = positions.shape[0]
        
        if self.depth_activation == 'sigmoid':
            pred_depth = torch.sigmoid(positions[...,:1])
            pred_depth = self.near_depth * (1 - pred_depth) + self.far_depth * pred_depth
        elif self.depth_activation == 'softplus':
            pred_depth = torch.nn.functional.softplus(positions[...,:1])
            pred_depth = torch.clamp(pred_depth,min=self.near_depth,max=self.far_depth)
        elif self.depth_activation == 'identity':
            pred_depth = positions[...,:1]
            pred_depth = torch.clamp(pred_depth,min=self.near_depth,max=self.far_depth)


        u = positions[...,1:2]
        v = positions[...,2:3]
        pixels_coords = torch.cat([u, v],dim=-1)
        
        pred_orientations_cam = rearrange(pred_orientations_cam, 'B N D -> (B N) D')
        pred_orientations_cam = rearrange(build_rotation(pred_orientations_cam), '(B N) h w -> B N h w', B=num_images)

        positions = []
        orientations = []
        for k in range(num_images):
            cam_gaussians = pred_to_cam_coords(pred_depth[k,...,0], pixels_coords[k], intrinsics[k], local_coords=False)
            
            if offset_3d is not None:
                cam_gaussians += offset_3d[k]

            if self.save_local_positions:
                self.cam_gaussians = cam_gaussians.detach()
            # Multiply with the inverse of extrinsic matrix to transform to world coordinates
            # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
            

            R_c2w = c2w_R[k]
            t_c2w = c2w_t[k]

            # Apply the rotation and translation to the camera coordinates
            world_coords_points = R_c2w[None] @ cam_gaussians[:,:,None] + t_c2w[None,:,None]
            #world_coords_points = torch.einsum("ij,hwj->hwi", R_c2w, cam_gaussians) + t_c2w
            positions.append(world_coords_points[...,0])
            #positions.append(cam_gaussians)
            
            world_orientations = R_c2w[None] @ pred_orientations_cam[k]
            world_orientations = extract_rotation(world_orientations)
            orientations.append(world_orientations)
        positions = torch.stack(positions,dim=0)
        orientations = torch.stack(orientations,dim=0)
        #orientations = rearrange(pred_orientations_cam,'B H W G D -> B (H W G) D')
        
        if need_reshape:
            positions = rearrange(positions, '(B V) N D -> B V N D', B=B, V=V)
            orientations = rearrange(orientations, '(B V) N D -> B V N D', B=B, V=V)
            pred_depth = rearrange(pred_depth, '(B V) N D -> B V N D', B=B, V=V)

        return positions, orientations, pred_depth

    def encoder_features_processing(self, features, images):
        if self.parent_model=='vggt':
            patch_start_idx = 5
        else:
            patch_start_idx = 17

        if self.features_mode == 'DPT':
            features, conf_map = self.features_head(aggregated_tokens_list=features, images=images, patch_start_idx=patch_start_idx)
            #features = self.features_norm(features)
            features = rearrange(features,'B V H W C -> (B V) C H W')
            self.conf_map = conf_map[:,:,None]
            conf_map = rearrange(self.conf_map,'B V C H W -> (B V) C H W')
        elif self.features_mode == 'depatchify':
            B, V, _, H, W = images.shape
            
            if 'vggt' in self.parent_model:
                features = torch.cat([self.norm(f[:, :, patch_start_idx:]) for f in features],dim=-1)
                features = rearrange(features,'B V N D -> (B V) N D')
            elif 'depthanything3' in self.parent_model:
                features = torch.cat([self.norm(f) for f in features],dim=-1)
                features = rearrange(features,'B V N D -> (B V) N D')
            elif self.parent_model=='mapanything':
                features = [self.norm1(features[0]), self.norm2(features[1]), self.norm2(features[2]), self.norm2(features[2])]
                features = torch.cat(features,dim=-1)
            
            features = self.depatchifier(features)
            features = rearrange(
                                            features,
                                            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                            h=H//self.patch_size,
                                            w=W//self.patch_size,
                                            p1=self.patch_size,
                                            p2=self.patch_size,
                                            c=self.num_features_channels
                                        )
            conf_map = None

        elif self.features_mode == 'zeros':
            B, V, _, H, W = images.shape
            features = torch.zeros((B*V, self.num_features_channels, H, W)).to(images.device)
            conf_map = None

        if self.conf_mode=='gausians':
            conf_map = None
        return features, conf_map

    def forward(self, images, features, depth, depth_conf, pointmap, point_conf, extrinsics, intrinsics):
        """
        images: tensor of shape B V 3 H W
        return
        gaussians: tensor of shape B V N D
        """
        B, V, _, H, W = images.shape
        self.num_images = V

        if self.features_mode is not None:
            features, conf_map_feat = self.encoder_features_processing(features, images)

        #images = rearrange(images,'B V C H W -> (B V) C H W')
        
        if pointmap is not None:
            pointmap = rearrange(pointmap,'B V C H W -> (B V) C H W')
            point_conf = rearrange(point_conf,'B V C H W -> (B V) C H W')
        else:
            depth = rearrange(depth,'B V C H W -> (B V) C H W')
            depth_conf = rearrange(depth_conf,'B V C H W -> (B V) C H W')

        x = self.image_normalizer(rearrange(images,'B V C H W -> (B V) C H W')).detach()
        if self.depth_input:
            if self.geometry_mode == 'pointmap':
                x = torch.cat([x, pointmap.detach(), point_conf.detach()],dim=1)
            else:
                x = torch.cat([x, depth.detach(), depth_conf.detach()],dim=1)
        
        if self.features_mode is not None:
            x = torch.cat([x, features],dim=1)
        
        x = self.backbone(x)

        if self.conf_mode == 'after_backbone':
            conf_map = torch.sigmoid(self.conf_head(x))
            self.conf_map = rearrange(conf_map,'(B V) C H W -> B V C H W', B=B, V=V)
        elif self.conf_mode == 'DPT':
            conf_map = conf_map_feat
        else:
            conf_map = None

        gaussians = self.features_to_gaussians(x, images, depth, depth_conf, pointmap, point_conf, conf_map)
        gaussians = rearrange(gaussians, '(B V) N D -> B V N D', B=B, V=V)
        with torch.amp.autocast('cuda', enabled=False):
            gaussians = self.compute_gaussians_attributes(gaussians, extrinsics, intrinsics)

        return gaussians

