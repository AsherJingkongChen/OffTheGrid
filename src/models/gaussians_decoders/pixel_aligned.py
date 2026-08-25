import torch
import numpy as np
from einops import rearrange
from src.utils.gaussians_utils import patchify, bilinear_interpolation, RGB2SH
from src.models.gaussians_decoders.base_decoder import GaussiansDecoder
from src.models.gaussians_decoders.layers import MLP

class PixelAlignedDecoder(GaussiansDecoder):
    def __init__(self, parent_model='vggt', geometry_mode='depth', patch_size=14,
                    depth_input=True, depth_mode='prior', features_mode="depatchify",color_mode='prior', gaussians_shape='balls', add_offset=False):
        
        if add_offset:
            color_mode=='predict'

        super().__init__(parent_model, geometry_mode, patch_size, depth_input, depth_mode, features_mode, color_mode, gaussians_shape)
        
        self.downscale_ratio = 2
        self.offset_3d = add_offset
        
        if self.downscale_ratio > 1:
            self.downscaler = torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.Conv2d(self.d, self.d,1))


        dim_gaussians_input = self.d + 1 + 3
        self.MLP_decoder = MLP(input_dim=dim_gaussians_input, output_dim=14 + self.predict_conf + 3*self.offset_3d, net_depth=2, net_width=128)
        self.relu = torch.nn.ReLU(inplace=True)

    def set_coords_grids(self, H, W):
        y_coords = torch.linspace(0, H - 1, H) + 0.5
        x_coords = torch.linspace(0, W - 1, W) + 0.5
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords_map = torch.stack([xx,yy],dim=0)[None]
        self.coords_grids = coords_map #B N 2 h w

    def features_to_gaussians(self, x, images, depth, depth_conf, pointmap, point_conf):
        B, V, _, H, W = images.shape
        self.set_coords_grids(H//self.downscale_ratio, W//self.downscale_ratio)
        images = rearrange(images,'B V C H W -> (B V) C H W')
        self.image_coordinates = rearrange(self.coords_grids, 'B D H W -> B (H W) D').expand(B*V,-1,-1).to(x.device)
        self.image_coordinates *= self.downscale_ratio
        if self.downscale_ratio == 1:
            gaussians_colors = rearrange(images,'B C H W -> B (H W) C')
            gaussians_depth = rearrange(depth,'B C H W -> B (H W) C')
        else:
            gaussians_colors = bilinear_interpolation(images,self.image_coordinates)
            gaussians_depth = bilinear_interpolation(depth,self.image_coordinates)

        if self.downscale_ratio > 1:
            x = self.downscaler(x)
        x = rearrange(x, 'B D H W -> B (H W) D')

        gaussians_tokens = torch.cat([x, gaussians_colors.detach(),gaussians_depth.detach()],dim=-1)
        gaussians = self.MLP_decoder(gaussians_tokens)
        
        gaussians[...,1:3] = self.image_coordinates
        if not self.predict_color:
            gaussians_colors = RGB2SH(gaussians_colors)
            gaussians[...,11:14] = gaussians_colors
        if self.geometry_mode == 'pointmap':
            gaussians[...,:3] = gaussians_point
            gaussians = torch.cat([gaussians,gaussians_point_conf],dim=-1)
        else:
            if self.depth_mode == 'refine':
                offset = (torch.sigmoid(0.01 * gaussians[...,:1]) - 0.5)*0.02
                gaussians[...,:1] = torch.clamp(gaussians_depth + offset,min=0.01)
            elif self.depth_mode == 'prior':
                gaussians[...,:1] = gaussians_depth
        return gaussians