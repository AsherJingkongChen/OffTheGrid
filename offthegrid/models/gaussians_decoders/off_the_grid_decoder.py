import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from offthegrid.utils.gaussians_utils import patchify, bilinear_interpolation, RGB2SH
from offthegrid.utils.adaptive_density_utils import get_adaptive_patches_from_images, visualize_adaptive_patches
from offthegrid.models.gaussians_decoders.layers import MLP
from offthegrid.models.gaussians_decoders.base_decoder import GaussiansDecoder

class OffTheGridDecoder(GaussiansDecoder):
    def __init__(self,
                parent_model='vggt',
                geometry_mode='depth',
                num_scales=3,
                num_gaussians_per_patch=[16,32,64],
                patch_size=14, 
                depth_input=True,
                depth_mode='prior',
                features_mode="depatchify",
                conf_mode='gaussians',
                color_mode='prior',
                gaussians_shape='disk'):
        
        super().__init__(parent_model, geometry_mode, patch_size, depth_input, depth_mode, features_mode, conf_mode, color_mode, gaussians_shape)

        #Adaptive density options
        self.num_scales = num_scales
        self.scales = range(self.num_scales)
        self.density_mode = "adaptive_image"
        if self.num_scales == 1:
            self.percentages = [1.]
        elif self.num_scales == 2:
            self.percentages = [0.66, 0.34]
        elif self.num_scales == 3:
            self.percentages = [0.55, 0.30, 0.15]
        else:
            raise NotImplementedError

        #Gaussians parameters decoder
        self.num_gaussians_per_patch = num_gaussians_per_patch
        if isinstance(self.num_gaussians_per_patch, int):
            self.num_gaussians_per_patch = [num_gaussians_per_patch]
        self.decoder_model='MLP'
        self.temperature = 1.0

        #Initialize per-density heads
        self.conv_det = nn.ModuleDict()
        self.conv_desc = nn.ModuleDict()
        for k, scale in enumerate(self.scales):
            num_gaussians_per_patch = self.num_gaussians_per_patch[k]
            latent_dim = self.d#max(self.d, num_gaussians_per_patch)
            self.conv_det[str(scale)] = nn.Sequential(
                torch.nn.Conv2d(self.d, latent_dim, kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                torch.nn.Conv2d(latent_dim, num_gaussians_per_patch , kernel_size=1, stride=1)
                )
            desc_out = self.d
            self.conv_desc[str(scale)] = nn.Sequential(
                nn.Conv2d(self.d, self.d, kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(self.d, self.d, kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(self.d, desc_out , kernel_size=1, stride=1)
                )
        
        self.softmax = torch.nn.Softmax(dim=1)
        dim_gaussians_input = self.d
        self.MLP_decoder = MLP(input_dim=dim_gaussians_input, output_dim=14 + self.predict_conf, net_depth=2, net_width=128)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def compute_coordinates_from_heatmaps(self, heatmaps):
        #DSNT operation
        coords_grid = self.coords_grids.to(heatmaps.device)
        points = (coords_grid * heatmaps[...,None,:,:]).sum(dim=(-1,-2))
        return points

    def features_to_gaussians(self, x, images, depth, depth_conf, pointmap, point_conf, conf_map):

        B, V, _, H, W = images.shape
        
        if self.num_scales>1:
            #adaptive_masks, self.num_patches = get_adaptive_patches_from_images(images, num_scales=len(self.scales))
            adaptive_masks, self.num_patches = get_adaptive_patches_from_images(images,mode=self.density_mode, conf=conf_map, num_scales=self.num_scales, percentages=self.percentages, patch_size=self.patch_size)
            if self.visualization_mode:
                self.patch_viz = visualize_adaptive_patches(images, adaptive_masks, patch_size=self.patch_size, alpha=0.3)
            
            num_gaussians = sum([self.num_patches[k] * self.num_gaussians_per_patch[k] for k in range(self.num_scales)])
            
            indexing_thresh = torch.tensor([0]+[self.num_patches[k] * self.num_gaussians_per_patch[k] for k in range(3)])
            indexing_thresh = torch.cumsum(indexing_thresh,dim=0).int().tolist()
        else:
            num_gaussians = self.num_gaussians_per_patch[0] * (H // self.patch_size) * (W // self.patch_size)
            adaptive_masks = {"0": []}
        images = rearrange(images,'B V C H W -> (B V) C H W')
        self.set_coords_grids(H, W)

        self.image_coordinates = torch.empty(B*V,num_gaussians,2).to(x.device)
        all_gaussians = []
        for s, scale in enumerate(self.scales):
            if adaptive_masks[str(scale)] != [] or self.num_scales==1:
                #Compute per scale detection and description features
                desc = self.conv_desc[str(scale)](x)
                det = self.conv_det[str(scale)](x)
                det = patchify(det, (self.patch_size,self.patch_size))

                #Compute gaussians 2D 'off the grid' positions
                det = rearrange(det, 'B P G h w -> (B P G) (h w)')
                #with torch.amp.autocast('cuda',enabled=False):
                det = det / self.temperature
                det = self.softmax(det)
                det = rearrange(det, '(B P G) (h w) -> (B G) P h w', h=self.patch_size, B=B*V, G=self.num_gaussians_per_patch[s])
                points_2d = self.compute_coordinates_from_heatmaps(det)
                image_coordinates = rearrange(points_2d, '(B G) P D -> B P G D', B=B*V, G=self.num_gaussians_per_patch[s])
                if len(self.scales)>1:                    
                    indices = adaptive_masks[str(scale)][:,:,None,None].expand(-1,-1,self.num_gaussians_per_patch[s],2).long()
                    selected_coordinates = torch.gather(image_coordinates, dim=1, index=indices)
                    selected_coordinates = rearrange(selected_coordinates, 'B P G D -> B (P G) D')

                    self.image_coordinates[:,indexing_thresh[s]:indexing_thresh[s+1]] = selected_coordinates
                else:
                    selected_coordinates = rearrange(image_coordinates, 'B N G D -> B (N G) D') 
                    self.image_coordinates = selected_coordinates

                #Extract a descriptor for each gaussian at detected 2D locations
                gaussians_desc = bilinear_interpolation(desc,selected_coordinates)


                if not self.predict_color:
                    gaussians_colors = bilinear_interpolation(images,selected_coordinates)
                    gaussians_colors = RGB2SH(gaussians_colors)
                
                if self.depth_mode != 'predict':
                    if pointmap is not None:
                        gaussians_point = bilinear_interpolation(pointmap,selected_coordinates)
                        gaussians_point_conf = bilinear_interpolation(point_conf,selected_coordinates)
                    else:
                        gaussians_depth = bilinear_interpolation(depth,selected_coordinates)

                if self.conf_mode != 'gaussians':
                    gaussians_conf = bilinear_interpolation(conf_map,selected_coordinates)
                #Compute primitives parameters from descriptors
                gaussians = self.MLP_decoder(gaussians_desc)


                gaussians[...,1:3] = selected_coordinates #Overwrite gaussian positions with detection
                if not self.predict_color:
                    gaussians[...,11:14] = gaussians_colors
                if self.conf_mode != 'gaussians':
                    gaussians[...,14:15] = gaussians_conf
                if self.geometry_mode == 'pointmap':
                    gaussians[...,:3] = gaussians_point
                else:
                    if self.depth_mode == 'refine':
                        offset = 0.1 * gaussians[...,:1]
                        #offset = (torch.sigmoid(gaussians[...,:1]) - 0.5)*0.02
                        gaussians[...,:1] = torch.clamp(gaussians_depth + offset,min=self.near_depth, max=self.far_depth)
                    elif self.depth_mode == 'prior':
                        gaussians[...,:1] = gaussians_depth
                all_gaussians.append(gaussians)
        all_gaussians = torch.cat(all_gaussians,dim=1)
        return all_gaussians

    

