import argparse
import os
import torch
import torchvision
import numpy as np
import time
from src.models.off_the_grid import OffTheGrid
from src.utils.os_utils import str2bool, tensor_to_mp4
from src.datasets.simple_inference_dataset import ImageFolderDataset
from src.utils.trajectory_utils import generate_trajectory, render_trajectory
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument("--input_dir", type=str, default='./', help="Input folder containing .jpg or .png images")
    parser.add_argument("--output_dir", type=str, default='./', help="Where to save the output files")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images per scene")
    parser.add_argument("--num_video_frames", type=int, default=500, help="Number of images per scene")
    parser.add_argument("--sampling_mode", type=str, default='linear', help=" ")
    parser.add_argument("--load_size", type=int, default=518, help=" ")
    parser.add_argument("--loading_mode", type=str, default='no_crop', help=" ")
    parser.add_argument("--trajectory_mode", type=str, default='spiral', help=" ")
    parser.add_argument("--ckpt_folder", type=str, default='/nfs/cvn/amoreau/checkpoints/foundationGS/final_model_thursday/', help=" ")
    parser.add_argument("--detector_type", type=str, default='offthegrid', help="decoder architecture")
    parser.add_argument("--save_ply", type=str2bool, default=True, help="save gaussian model as .ply file for export in SuperSplat")
    parser.add_argument("--upscale_ratio", type=float, default=1.0, help="")
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    dataset = ImageFolderDataset(args.input_dir, args.num_images, args.sampling_mode, args.loading_mode)

    model = OffTheGrid(encoder='vggt', decoder=args.detector_type, geometry_mode='depth', num_gaussians_per_patch=[16,32,64], predict_conf=True, use_features=True)
    model.load_from_checkpoint(args.ckpt_folder)
    model.eval() #Increases VRAM comsumption need to figure out why

    data = dataset.__getitem__(0)
    start_time = time.time()
    H,W = data["images"].shape[-2:]
    with torch.no_grad():
        gaussians, metadata = model(data["images"].to(device))
    print(f"Inference time: {time.time() - start_time} seconds.")

    os.makedirs(os.path.join(args.output_dir,'inputs'),exist_ok=True)
    for k in range(args.num_images):
        torchvision.utils.save_image(data["images"][0,k].cpu(), os.path.join(args.output_dir,f'inputs', f"input_{str(k).zfill(2)}.png"))

    if args.save_ply:
        path = os.path.join(args.output_dir,f'gaussians.ply')
        model.save_gaussians(path,gaussians,batch_idx=0, conf_fusion=True,save_each_frame=False)

    reference_depth = metadata["depths"][0,0,0] #H W

    traj_modes = ['spiral','bullet_time_linear','bullet_time_linear_zoom', 'bullet_time_circle', 'bullet_time_loop', 'bullet_time_dolly_zoom', 'bullet_time_sway', 'bullet_time_wobble']
    #traj_modes = ['interpolation+spiral']
    time_modes = ['linear']#,'ease_in','ease_out', 'smoothstep', 'cos_ease']
    for mode in traj_modes:
        for time_mode in time_modes:
            trajectory, intr_traj = generate_trajectory(args.num_video_frames, mode, metadata["extrinsics"][0], metadata["intrinsics"][0].mean(dim=0), reference_depth, time_warp_mode=time_mode)
            video_tensor, depth_tensor = render_trajectory(model, gaussians, trajectory, intr_traj, args.num_video_frames, H=H, W=W, bg_color=torch.tensor([0.,0.,0.]).to(device), upscale_ratio=args.upscale_ratio)
            tensor_to_mp4(video_tensor,os.path.join(args.output_dir, f'{mode}_{time_mode}.mp4'),fps=24)
            #tensor_to_mp4(depth_tensor,os.path.join(args.output_dir, f'{mode}_depth.mp4'),fps=24)

