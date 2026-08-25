import argparse
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
from src.models.off_the_grid import OffTheGrid
from src.utils.trajectory_utils import interpolate_extrinsics, generate_spiral_path
from src.utils.os_utils import str2bool, tensor_to_mp4, save_camera_poses
from src.utils.colmap_utils import export_to_colmap_txt
from src.utils.gaussians_utils import SH2RGB
from src.datasets.simple_inference_dataset import ImageFolderDataset
from src.utils.trajectory_utils import generate_trajectory, render_trajectory

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument("--input_dir", type=str, default='./', help="Input folder containing .jpg or .png images")
    parser.add_argument("--output_dir", type=str, default='./', help="Where to save the output files")
    parser.add_argument("--num_images", type=int, default=8, help="Number of images per scene")
    parser.add_argument("--sampling_mode", type=str, default='linear', help=" ")
    parser.add_argument("--loading_mode", type=str, default='no_crop', help="center_crop, vggt_crop, vggt_pad, resize, closest")
    parser.add_argument("--load_size", type=int, default=518, help="height of source images")
    parser.add_argument("--density_mode", type=str, default='multi_res_image', help="multi_res_image, low, medium, high")
    parser.add_argument("--encoder_model", type=str, default='vggt', help="vggt, depthanything3")
    parser.add_argument("--ckpt_folder", type=str, default='./checkpoints/offthegrid/', help=" ")
    parser.add_argument("--decoder_type", type=str, default='multires_detector', help="decoder architecture")
    parser.add_argument("--save_ply", type=str2bool, default=True, help="save gaussian model as .ply file for export in SuperSplat")
    parser.add_argument("--export_to_colmap", type=str2bool, default=True, help="save 3D reconstruction as a COLMAP model")
    parser.add_argument("--save_geometry", type=str2bool, default=True, help="save gaussian model as .ply file for export in SuperSplat")
    parser.add_argument("--visualize_adaptive_density", type=str2bool, default=False, help="Save an image showing the multi-density patch classification")
    parser.add_argument("--shared_intrinsics", type=str2bool, default=False, help="average intrinsics parameters over images")
    parser.add_argument("--render_videos", type=str2bool, default=True, help="average intrinsics parameters over images")
    parser.add_argument("--video_trajectory", type=str, default='interpolation+spiral', help="camera trajectory in the video")
    parser.add_argument("--num_video_frames", type=int, default=200, help="number of iteration for test-time optimization")
    parser.add_argument("--test_time_optimization", type=str2bool, default=False, help="")
    parser.add_argument("--num_iterations_tto", type=int, default=10, help="number of iteration for test-time optimization")
    parser.add_argument('--num_gaussians_per_patch', type=int, nargs='*', help='List that defines number of primitives per patch for each density level', required=False)
    parser.add_argument("--export_mesh", type=str2bool, default=False, help="")
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)
    
    dataset = ImageFolderDataset(args.input_dir, args.num_images, args.sampling_mode, args.loading_mode, args.load_size)

    if args.num_gaussians_per_patch == []:
        args.num_gaussians_per_patch = [16,32,64]
    
    
    model = OffTheGrid(encoder=args.encoder_model, decoder=args.decoder_type, geometry_mode='depth', num_gaussians_per_patch=args.num_gaussians_per_patch, predict_conf=True, use_features=True)
    model.load_from_checkpoint(args.ckpt_folder)
    model.eval() #Increases VRAM comsumption need to figure out why

    model.gaussians_decoder.density_mode = args.density_mode
    if args.visualize_adaptive_density:
        model.gaussians_decoder.visualization_mode=True

    data = dataset.__getitem__(0)
    B, N, C, H, W = data["images"].shape
    start_time = time.time()
    with torch.no_grad():
        gaussians, metadata = model(data["images"].to(device))
    print(f"Inference time: {time.time() - start_time} seconds.")
    if args.test_time_optimization:
        raise NotImplementedError

    os.makedirs(os.path.join(args.output_dir,'inputs'),exist_ok=True)
    for k in range(args.num_images):
        torchvision.utils.save_image(data["images"][0,k].cpu(), os.path.join(args.output_dir,f'inputs', f"input_{str(k).zfill(2)}.png"))

    if args.save_ply:
        path = os.path.join(args.output_dir,f'gaussians.ply')
        model.save_gaussians(path,gaussians,batch_idx=0, conf_fusion=True,save_each_frame=False)

    if args.visualize_adaptive_density:
        if model.gaussians_decoder.num_scales > 1:
            im_viz = model.gaussians_decoder.patch_viz
            torchvision.utils.save_image(im_viz[0], os.path.join(args.output_dir,f'adaptive_density_{args.density_mode}.png'))

    pred_depths = metadata["depths"][0].detach().cpu().numpy()#.permute(0,2, 3, 1).numpy()
    vmin = np.min(pred_depths)
    pred_depths -= vmin
    vmax = np.max(pred_depths)
    pred_depths /= vmax

    if args.save_geometry:
        #save camera parameters
        cameras_params_file = os.path.join(args.output_dir,'cameras.json')
        save_camera_poses(metadata["extrinsics"][0], metadata["intrinsics"][0], data["paths"], cameras_params_file)

        #save depth maps
        os.makedirs(os.path.join(args.output_dir,'depths'),exist_ok=True)
        for k in range(args.num_images):
            depth_visu = plt.cm.magma_r(pred_depths[k,0])[..., :3]
            depth_visu = torch.from_numpy(depth_visu).permute(2, 0, 1).float()
            torchvision.utils.save_image(depth_visu, os.path.join(args.output_dir,'depths', f'depth_{str(k).zfill(2)}' + ".png"))

    if args.export_to_colmap:
        w2c_extrinsics = metadata["extrinsics"][0] #Shape N 4 4 
        intrinsics = metadata["intrinsics"][0] #Shape N 3 3
        pruned_gaussians = model.confidence_fusion(gaussians,0)
        points3D = pruned_gaussians['means']#.reshape(-1,3)
        colors = (SH2RGB(pruned_gaussians['features_dc']) * 255).int()
        image_names = data["paths"]
        print("Exporting to COLMAP format...")
        export_to_colmap_txt(
            args.output_dir, 
            w2c_extrinsics, 
            intrinsics, 
            points3D, 
            colors,
            image_names
        )
    

    if args.render_videos:
        trajectory, intr_traj = generate_trajectory(args.num_video_frames, args.video_trajectory, metadata["extrinsics"][0], metadata["intrinsics"][0].mean(dim=0), time_warp_mode='linear')
        video_tensor, depth_tensor = render_trajectory(model, gaussians, trajectory, intr_traj, 200, H=H, W=W, bg_color=torch.tensor([0.,0.,0.]).to(device))
        tensor_to_mp4(video_tensor,os.path.join(args.output_dir, f'render_rgb.mp4'),fps=20)
        tensor_to_mp4(depth_tensor,os.path.join(args.output_dir, f'render_depth.mp4'),fps=20)

    if args.export_mesh:
        from src.utils.mesh_utils import tsdf_fusion
        mesh_mode = 'pred'
        if mesh_mode == 'render':
            renders, depths, alphas = [], [], []
            bg_color=torch.tensor([0.5,0.5,0.5]).to(device)
            for k in range(args.num_images):
                with torch.no_grad():
                    render, depth, normals, alpha = model.render_image(gaussians, metadata['extrinsics'][0, k], metadata['intrinsics'][0, k], H, W, conf_fusion=model.predict_conf, batch_idx=0, render_depth_normal=True, bg_color=bg_color)
                    renders.append(render)
                    depths.append(depth)
                    alphas.append(alpha)
        else:
            renders = [data["images"][0,k] for k in range(args.num_images)]
            depths = [metadata["depths"][0,k] for k in range(args.num_images)]
            alphas = [torch.ones_like(depths[0][0]) for k in range(args.num_images)]

        tsdf_fusion(renders, depths, alphas, metadata['extrinsics'][0],
                    metadata['intrinsics'][0], args.num_images, W, H,
                    args.output_dir,  cuda_device='CPU:0', quantile=1.0)
