import os
import torch
import torchvision
import numpy as np
import argparse
import time
import json
import cv2
import pandas as pd
import csv

from src.models.off_the_grid import OffTheGrid
from src.models.baselines import DA3GaussianModel
from src.utils.geometry_utils import closed_form_inverse_se3, matrix_to_quaternion, quaternion_to_matrix
from src.utils.metrics_utils import compute_rendering_metrics, evaluate_intrinsics
from src.utils.os_utils import str2bool, tensor_to_mp4, save_camera_poses, save_depth_map
from src.utils.eval_utils import cam_eval, depth_eval
from src.utils.trajectory_utils import interpolate_extrinsics, generate_spiral_path
from src.utils.alignment_utils import camera_pose_optimization_on_test_images
from src.datasets.Charge_dataset import ChargeDataset
from src.datasets.scrream_dataset import SCRREAMDataset
from src.datasets.mipnerf360_dataset import MipNeRF360Dataset
from src.datasets.tanks_and_temples_dataset import TanksAndTemplesDataset
from src.datasets.DL3DV_benchmark_dataset import DL3DVBenchmarkDataset
from src.datasets.sevenscenes_dataset import SevenScenesDataset
from src.datasets.pixelsplat_datasets import PixelSplatDataset
from src.datasets.realX3D_dataset import RealX3D_Dataset
from src.datasets.robustnerf_dataset import RobustNerf_Dataset
from src.datasets.nerfonthego_dataset import NerfOnTheGo_Dataset
from src.datasets.phototourism_dataset import Phototourism_Dataset
from src.datasets.minimal_dataset import MinimalDataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def define_dataset(dataset_name, num_images_per_scene, args):
    if dataset_name == 'DL3DV':
        dataset = DL3DVBenchmarkDataset(num_images_per_scene, num_images_per_scene-1,
                                    loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == '7scenes':
        dataset = SevenScenesDataset(num_images_per_scene, num_images_per_scene-1,
                                     loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'Charge':
        if num_images_per_scene==3:
            input_mode = 'sparse_3'
        elif num_images_per_scene==6:
            input_mode = 'sparse_6'
        elif num_images_per_scene==9:
            input_mode = 'sparse_9'
        elif num_images_per_scene==25:
            input_mode = 'dense'
        dataset = ChargeDataset(input_mode=input_mode, num_scenes_per_seq=5, 
                                loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'SCRREAM':
        dataset = SCRREAMDataset(num_images_per_scene, num_images_per_scene-1, 
                                loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'TanksandTemples':
        dataset = TanksAndTemplesDataset(num_images_per_scene, num_images_per_scene-1, 
                            split='all', loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'mipnerf360':
        dataset = MipNeRF360Dataset(num_images_per_scene, num_images_per_scene-1,
                                 loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'RealX3D':
        dataset = RealX3D_Dataset(num_images_per_scene, loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'RobustNeRF':
        dataset = RobustNerf_Dataset(num_images_per_scene, loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'nerfonthego':
        dataset = NerfOnTheGo_Dataset(num_images_per_scene, loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'phototourism':
        dataset = Phototourism_Dataset(num_images_per_scene, loading_mode=args.loading_mode, load_size=args.load_size)
    elif dataset_name == 'minimal_dataset':
        dataset = MinimalDataset(num_images_per_scene, loading_mode=args.loading_mode, load_size=args.load_size)
    else:
        raise NotImplementedError
    return dataset

def evaluate(args, model, dataset_name, num_images_per_scene, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = define_dataset(dataset_name, num_images_per_scene, args)
    num_scenes = len(dataset)
    conf_fusion = model.predict_conf
    bg_color=torch.tensor([0.5,0.5,0.5]).to(device)

    # Initialize metric dicts
    metrics = {"train_PSNR": [], "train_SSIM": [], "train_LPIPS": [], "test_PSNR": [], "test_SSIM": [], "test_LPIPS": []}
    fov_metrics = {"intrinsics_MAE": [], "intrinsics_angular_error": [], "T@3": []}
    cpe_metrics = {"R@5": [], "R@10": [], "R@15": [], "R@20": [], "R@30": [], "T@5": [], "T@10": [], "T@15": [], "T@20": [], "T@30": [], "AUC@5": [], "AUC@10": [], "AUC@15": [], "AUC@20": [], "AUC@30": []}
    depth_metrics = {"AbsRel": [], "SqRel": [], "RMSElinear": [], "RMSElog": [], "InvRMSE": [], "SILogRMSE": [], "d1": [], "d2": [], "d3": []}
    depth_metrics_perframe = {k: [] for k in depth_metrics.keys()}
    rendered_depth_metrics = {k: [] for k in depth_metrics.keys()}
    primitives_metrics = {"num_gaussians": [], "num_gaussians_perpixel": []}

    for scene in range(num_scenes):
        scene_dir = os.path.join(output_dir, f'scene_{scene}')
        os.makedirs(scene_dir, exist_ok=True)
        
        batch = dataset.__getitem__(scene)
        images = batch["images"][None].cuda()
        
        with torch.no_grad():
            gaussians_attributes, data = model(images)
        
        extrinsic = data["extrinsics"].cuda()
        intrinsic = data["intrinsics"].cuda()
        if args.shared_intrinsics:
            fx = intrinsic[..., 0, 0].mean(dim=1,keepdim=True)
            fy = intrinsic[..., 1, 1].mean(dim=1,keepdim=True)
            intrinsic[..., 0, 0] = fx
            intrinsic[..., 1, 1] = fy

        depth_map = data["depths"].cuda()
        depth_conf = data["depth_conf"].cuda()
        embedding = [a.cuda() for a in data["embeddings"]] if "embeddings" in data else None
        
        H, W = images.shape[-2], images.shape[-1]
        num_images = images.shape[1]
        
        if args.count_primitives:
            pruned_gaussians = model.confidence_fusion(gaussians_attributes,0)
            num_gaussians = pruned_gaussians['means'].shape[0]
            primitives_metrics["num_gaussians"].append(num_gaussians)
            primitives_metrics["num_gaussians_perpixel"].append(num_gaussians/(H * W * num_images))

        if args.evaluate_intrinsics and 'intrinsics' in batch:
            intr_metrics = evaluate_intrinsics(intrinsic, batch["intrinsics"])
            for met in intr_metrics:
                fov_metrics[met] += intr_metrics[met]

        if args.evaluate_camera_pose and 'extrinsics' in batch:
            pred_poses = torch.zeros(num_images_per_scene,4,4)
            pred_poses[:,:3] = data["extrinsics"][0]
            pred_poses = closed_form_inverse_se3(pred_poses)
            gt_poses = closed_form_inverse_se3(batch["extrinsics"])
            cpe = cam_eval(pred_poses, gt_poses)
            for met in cpe: cpe_metrics[met].append(cpe[met])
            
        if args.render_test_images:
            with torch.no_grad():
                target_data = model.align_target_views(images, batch["target_views"][None].cuda(), data['extrinsics'], scale_align=args.scale_align)
                aligned_test_extrinsics = target_data['extrinsics'][0].float().to(data['extrinsics'].device)
                target_data = {}
                num_test_cams = aligned_test_extrinsics.shape[0]
                test_intrinsics = intrinsic[0].mean(dim=0)[None].repeat(num_test_cams,1,1).cuda() # default to 'mean' intr mode
            if args.test_camera_pose_optimization: 
                try:
                    aligned_test_extrinsics = camera_pose_optimization_on_test_images(batch["target_views"][None].cuda(), gaussians_attributes, aligned_test_extrinsics[None], test_intrinsics[None], model)
                except:
                    pass

        if args.save_ply:
            model.save_gaussians(os.path.join(scene_dir, f'scene_{scene}_gaussians.ply'), gaussians_attributes, batch_idx=0, conf_fusion=conf_fusion, save_each_frame=False)

        if args.save_camera_params:
            save_camera_poses(data["extrinsics"][0], data["intrinsics"][0], batch["train_paths"], os.path.join(scene_dir, 'cameras.json'))
            if args.render_test_images:
                save_camera_poses(aligned_test_extrinsics, test_intrinsics, batch["test_paths"], os.path.join(scene_dir, 'target_cameras.json'))
        
        if args.test_time_optimization:
            model.use_normals_loss = False
            model.use_depth_loss = True
            start_time = time.time()
            gaussians_attributes, extrinsic, intrinsic = model.test_time_optimization(gaussians_attributes, images, depth_map, extrinsic, intrinsic, num_iterations=args.num_iterations_tto, params_to_optimize=['extrinsics'], shared_intrinsics=args.shared_intrinsics)
            if args.save_ply:
                model.save_gaussians(os.path.join(scene_dir, f'scene_{scene}_gaussians_tto.ply'), gaussians_attributes, batch_idx=0, conf_fusion=conf_fusion, save_each_frame=False)
        
        # Rendering evaluations
        if args.render_train_images or args.evaluate_rendered_depth:
            os.makedirs(os.path.join(scene_dir, 'train'), exist_ok=True)
            renders, source_rendered_depths = [], []

            for k in range(num_images):
                with torch.no_grad():
                    render, depth, normals, alphas = model.render_image(gaussians_attributes, extrinsic[0,k], intrinsic[0,k], H, W, conf_fusion=conf_fusion, batch_idx=0, render_depth_normal=True, bg_color=bg_color)
                if args.render_train_images:
                    torchvision.utils.save_image(render.cpu(), os.path.join(scene_dir, 'train', f'render_{k}.png'))
                    torchvision.utils.save_image(images[0,k].cpu(), os.path.join(scene_dir, 'train', f'gt_{k}.png'))
                if args.evaluate_rendering: renders.append(render)
                if args.evaluate_rendered_depth: source_rendered_depths.append(depth.detach().cpu())

            if args.evaluate_rendering:
                renders = torch.stack(renders,dim=0)
                psnr, ssim, lpips = compute_rendering_metrics(renders.detach().cpu(), images.cpu()[0])
                metrics["train_PSNR"].append(psnr.mean().item())
                metrics["train_SSIM"].append(ssim.mean().item())
                metrics["train_LPIPS"].append(lpips.mean().item())

        if args.render_test_images:
            os.makedirs(os.path.join(scene_dir, 'test'), exist_ok=True)
            if args.target_size != 518:
                H_test = W_test = args.target_size
                render_intrinsics = torch.zeros_like(test_intrinsics)
                render_intrinsics[:, 0, 0] = test_intrinsics[:, 0, 0] * (args.target_size / 518)
                render_intrinsics[:, 1, 1] = test_intrinsics[:, 1, 1] * (args.target_size / 518)
                render_intrinsics[:, 0, 2] = render_intrinsics[:, 1, 2] = args.target_size / 2
                test_images = batch["target_views_original"][None].cuda()
            else:
                H_test, W_test = H, W
                test_images = batch["target_views"][None].cuda()
                render_intrinsics = test_intrinsics
            
            renders = []      
            for k in range(test_images.shape[1]):
                with torch.no_grad():
                    render, depth, normals, alphas = model.render_image(gaussians_attributes, aligned_test_extrinsics[k], render_intrinsics[k], H_test, W_test, conf_fusion=conf_fusion, batch_idx=0, render_depth_normal=True, bg_color=bg_color)
                if render is None: render = torch.ones_like(test_images[0,k]) * 0.5
                render[:,torch.sum(render,dim=0)==0] = 0.5 

                torchvision.utils.save_image(render.cpu(), os.path.join(scene_dir, 'test', f'render_{k}.png'))
                torchvision.utils.save_image(test_images[0,k].cpu(), os.path.join(scene_dir, 'test', f'gt_{k}.png'))
                if args.evaluate_rendering: renders.append(render)

            if args.evaluate_rendering:
                renders = torch.stack(renders,dim=0).cpu()
                psnr, ssim, lpips = compute_rendering_metrics(renders.detach(), test_images[0].cpu())
                metrics["test_PSNR"].append(psnr.mean().item())
                metrics["test_SSIM"].append(ssim.mean().item())
                metrics["test_LPIPS"].append(lpips.mean().item())
        
        if args.save_depth:
            os.makedirs(os.path.join(scene_dir, 'depths'), exist_ok=True)

        if args.evaluate_depth and 'depths_gt' in batch:
            pred_depths = depth_map[0].detach().cpu()
            gt_depth = batch["depths_gt"].cpu()
            depth_scene_metrics, depths_map_aligned = depth_eval(pred_depths, gt_depth, return_aligned=True, scene_alignment=True)
            depth_frame_metrics, depths_map_aligned_perframe = depth_eval(pred_depths, gt_depth, return_aligned=True, scene_alignment=False)
            for met in depth_metrics:
                depth_metrics[met].append(depth_scene_metrics[met])
                depth_metrics_perframe[met].append(depth_frame_metrics[met])
            if args.save_depth:
                vmin, vmax = torch.min(gt_depth), torch.max(gt_depth - torch.min(gt_depth))
                save_depth_map(gt_depth, os.path.join(scene_dir, 'depths', 'gt_depths.png'), vmin, vmax)
                save_depth_map(depths_map_aligned[:,None], os.path.join(scene_dir, 'depths', 'depths_scene_aligned.png'), vmin, vmax)
                save_depth_map(depths_map_aligned_perframe[:,None], os.path.join(scene_dir, 'depths', 'depths_frame_aligned.png'), vmin, vmax)

        if args.evaluate_rendered_depth and 'depths_gt' in batch:
            source_rendered_depths = torch.stack(source_rendered_depths,dim=0).detach().cpu()
            render_depth_scene_metrics, rend_depths_map_aligned = depth_eval(source_rendered_depths, gt_depth, return_aligned=True, scene_alignment=True)
            for met in rendered_depth_metrics: rendered_depth_metrics[met].append(render_depth_scene_metrics[met])
            if args.save_depth:
                save_depth_map(rend_depths_map_aligned, os.path.join(scene_dir, 'depths', 'render_depths_scene_aligned.png'), vmin, vmax)

        if args.render_videos:
            num_interpolated_frames = 30 * num_images
            interpolated_extrinsics = interpolate_extrinsics(extrinsic[0,:,:3].cpu(), num_interpolated_frames)
            if args.extrapolated_videos:
                num_frames_ellipse = 100
                num_video_frames = num_frames_ellipse + num_interpolated_frames
                extrap_poses = torch.from_numpy(generate_spiral_path(interpolated_extrinsics[num_interpolated_frames//2].cpu().numpy(), N=num_frames_ellipse, radius=0.1))
                trajectory = torch.cat([interpolated_extrinsics[:num_interpolated_frames//2], extrap_poses, interpolated_extrinsics[num_interpolated_frames//2:]], dim=0)
            else:
                num_video_frames = num_interpolated_frames
                trajectory = interpolated_extrinsics
            
            with torch.no_grad():
                video_tensor = torch.zeros(num_video_frames,3,H,W)
                for i in range(num_video_frames):
                    render, depth, normals, alphas = model.render_image(gaussians_attributes, trajectory[i].cuda(), intrinsic[0,0], H, W, conf_fusion=conf_fusion, batch_idx=0, bg_color=bg_color)
                    if render is not None:
                        render[:,torch.sum(render,dim=0)==0] = 0.5 
                        video_tensor[i] = render.cpu()
                tensor_to_mp4(video_tensor, os.path.join(output_dir, f'render_{scene}.mp4'), fps=20)

    if args.count_primitives:
        with open(os.path.join(output_dir,'primitives_count.txt'), "w") as f:
            f.write("Number of 3D Gaussians primitives after pruning \n")
            f.write(f"Computed on {num_scenes} scenes with {num_images_per_scene} images.\n")
            for met in primitives_metrics:
                if primitives_metrics[met]: # added check to prevent div by zero
                    dataset_average = sum(primitives_metrics[met]) / len(primitives_metrics[met])
                    f.write(f"{met} : {dataset_average} \n")

    if args.evaluate_camera_pose:        
        with open(os.path.join(output_dir,'camera_pose_estimation.txt'), "w") as f:
            f.write("Camera pose estimation metrics \n")
            f.write(f"Computed on {num_scenes} scenes with {num_images_per_scene} images.\n")
            for met in cpe_metrics:
                if cpe_metrics[met]:
                    dataset_average = sum(cpe_metrics[met]) / len(cpe_metrics[met])
                    f.write(f"{met} : {dataset_average} \n")

    if args.evaluate_depth:
        with open(os.path.join(output_dir,'depth_estimation.txt'), "w") as f:
            f.write("Depth estimation metrics \n")
            f.write(f"Computed on {num_scenes} scenes with {num_images_per_scene} images.\n\n")
            f.write("Per scene alignment \n")
            for met in depth_metrics:
                if depth_metrics[met]:
                    dataset_average = sum(depth_metrics[met]) / len(depth_metrics[met])
                    f.write(f"{met} : {dataset_average} \n")
            f.write("\nPer frame alignment \n")
            for met in depth_metrics_perframe:
                if depth_metrics_perframe[met]:
                    dataset_average = sum(depth_metrics_perframe[met]) / len(depth_metrics_perframe[met])
                    f.write(f"{met} : {dataset_average} \n")

    if args.evaluate_rendered_depth:
        with open(os.path.join(output_dir,'rendered_depth_estimation.txt'), "w") as f:
            f.write("Depth estimation metrics (rendered) \n")
            f.write(f"Computed on {num_scenes} scenes with {num_images_per_scene} images.\n")
            for met in rendered_depth_metrics:
                if rendered_depth_metrics[met]:
                    dataset_average = sum(rendered_depth_metrics[met]) / len(rendered_depth_metrics[met])
                    f.write(f"{met} : {dataset_average} \n")
    
    if args.evaluate_intrinsics:
         with open(os.path.join(output_dir,'intrinsics_metrics.txt'), "w") as f:
            if fov_metrics["intrinsics_MAE"]:
                dataset_mae = sum(fov_metrics["intrinsics_MAE"]) / len(fov_metrics["intrinsics_MAE"])
                dataset_angular_error = sum(fov_metrics["intrinsics_angular_error"]) / len(fov_metrics["intrinsics_angular_error"])
                dataset_thresh = sum(fov_metrics["T@3"]) / len(fov_metrics["T@3"])
                f.write(f"Intrinsics metrics:\n")
                f.write(f"MAE:{dataset_mae}, Angular error: {dataset_angular_error}, T@3: {dataset_thresh}\n\n")
                for s in range(len(fov_metrics["intrinsics_MAE"])):
                    f.write(f"Scene {s}, intrinsics, MAE :{fov_metrics['intrinsics_MAE'][s]}, angular error:{fov_metrics['intrinsics_angular_error'][s]}\n")

    if args.evaluate_rendering:
        with open(os.path.join(output_dir,'rendering_metrics.txt'), "w") as f:
            if args.render_test_images and metrics["test_PSNR"]:
                dataset_psnr = sum(metrics["test_PSNR"]) / len(metrics["test_PSNR"])
                dataset_ssim = sum(metrics["test_SSIM"]) / len(metrics["test_SSIM"])
                dataset_lpips = sum(metrics["test_LPIPS"]) / len(metrics["test_LPIPS"])
                f.write(f"Render metrics:\n")
                f.write(f"test images, PSNR:{dataset_psnr}, SSIM: {dataset_ssim}, LPIPS {dataset_lpips}\n\n")
                for s in range(len(metrics["test_PSNR"])):
                    f.write(f"Scene {s}, test images, PSNR :{metrics['test_PSNR'][s]}, SSIM: {metrics['test_SSIM'][s]}, LPIPS: {metrics['test_LPIPS'][s]}\n")
                f.write("\n")
            if args.render_train_images and metrics["train_PSNR"]:
                dataset_psnr = sum(metrics["train_PSNR"]) / len(metrics["train_PSNR"])
                dataset_ssim = sum(metrics["train_SSIM"]) / len(metrics["train_SSIM"])
                dataset_lpips = sum(metrics["train_LPIPS"]) / len(metrics["train_LPIPS"])
                f.write(f"Render metrics:\n")
                f.write(f"train images, PSNR:{dataset_psnr}, SSIM: {dataset_ssim}, LPIPS {dataset_lpips}\n\n")
                for s in range(len(metrics["train_PSNR"])):
                    f.write(f"Scene {s}, train images, PSNR :{metrics['train_PSNR'][s]}, SSIM: {metrics['train_SSIM'][s]}, LPIPS: {metrics['train_LPIPS'][s]}\n")

    avg_results = {
        'PSNR': np.mean(metrics["test_PSNR"]) if metrics["test_PSNR"] else None,
        'SSIM': np.mean(metrics["test_SSIM"]) if metrics["test_SSIM"] else None,
        'LPIPS': np.mean(metrics["test_LPIPS"]) if metrics["test_LPIPS"] else None,
        'train_PSNR': np.mean(metrics["train_PSNR"]) if args.render_train_images and metrics["train_PSNR"] else None,
        'train_SSIM': np.mean(metrics["train_SSIM"]) if args.render_train_images and metrics["train_SSIM"] else None,
        'train_LPIPS': np.mean(metrics["train_LPIPS"]) if args.render_train_images and metrics["train_LPIPS"] else None,
        'num_gaussians_perpixel': np.mean(primitives_metrics["num_gaussians_perpixel"]) if primitives_metrics["num_gaussians_perpixel"] else None,
        'AbsRel': np.mean(depth_metrics["AbsRel"]) if depth_metrics["AbsRel"] else None,
        'AUC@30': np.mean(cpe_metrics["AUC@30"]) if cpe_metrics["AUC@30"] else None,
        'angular_error': np.mean(fov_metrics["intrinsics_angular_error"]) if fov_metrics["intrinsics_angular_error"] else None
    }
    return avg_results

def save_results_tables(all_results, output_dir):
    # Flatten results into a list of records
    records = []
    for d_name, views_dict in all_results.items():
        for v_num, metrics in views_dict.items():
            row = {'Dataset': d_name, 'Views': v_num}
            row.update(metrics)
            records.append(row)
    
    df = pd.DataFrame(records)
    exp_name = os.path.basename(os.path.normpath(output_dir))
    
    render_cols = ['PSNR', 'SSIM', 'LPIPS']
    train_render_cols = ['train_PSNR', 'train_SSIM', 'train_LPIPS']
    geom_cols = ['AbsRel', 'AUC@30', 'angular_error']

    # 4.1 Per dataset table (Averaged across views configs)
    for dataset in df['Dataset'].unique():
        ds_df = df[df['Dataset'] == dataset]
        ds_dir = os.path.join(output_dir, dataset)
        os.makedirs(ds_dir, exist_ok=True)
        
        write_custom_csv(ds_df, 'Views', render_cols, exp_name, os.path.join(ds_dir, f'{exp_name}_metrics_rendering.csv'), is_render=True)
        write_custom_csv(ds_df, 'Views', train_render_cols, exp_name, os.path.join(ds_dir, f'{exp_name}_metrics_train_rendering.csv'), is_render=True)
        write_custom_csv(ds_df, 'Views', geom_cols, exp_name, os.path.join(ds_dir, f'{exp_name}_metrics_geometry.csv'), is_render=False)

    # 4.2 Global results averaged over datasets
    write_custom_csv(df, 'Dataset', render_cols, exp_name, os.path.join(output_dir, f'{exp_name}_global_metrics_rendering.csv'), is_render=True)
    write_custom_csv(df, 'Dataset', train_render_cols, exp_name, os.path.join(output_dir, f'{exp_name}_global_metrics_train_rendering.csv'), is_render=True)
    write_custom_csv(df, 'Dataset', geom_cols, exp_name, os.path.join(output_dir, f'{exp_name}_global_metrics_geometry.csv'), is_render=False)

    # 4.3 Results depending on number of views averaged across datasets
    write_custom_csv(df, 'Views', render_cols, exp_name, os.path.join(output_dir, f'{exp_name}_views_metrics_rendering.csv'), is_render=True)
    write_custom_csv(df, 'Views', train_render_cols, exp_name, os.path.join(output_dir, f'{exp_name}_views_metrics_train_rendering.csv'), is_render=True)
    write_custom_csv(df, 'Views', geom_cols, exp_name, os.path.join(output_dir, f'{exp_name}_views_metrics_geometry.csv'), is_render=False)

# Formatting helper for precision
def fmt(m, v):
    if pd.isna(v) or v is None: return ""
    if 'PSNR' in m or 'angular_error' in m: return f"{v:.2f}"
    return f"{v:.4f}"

# Custom CSV writer to match the exact requested Excel format
def write_custom_csv(sub_df, pivot_col, metrics, exp_name, filepath, is_render=False):
    if not all(c in sub_df.columns for c in metrics) or sub_df[metrics].dropna(how='all').empty:
        return

    # Group by pivot_col (Dataset or Views)
    df_group = sub_df.groupby(pivot_col)[metrics].mean()
    categories = sorted(df_group.index.tolist())
    
    # Averages across the grouped categories
    avg_row = df_group.mean()
    
    # Handle #G / im insertion
    has_g = is_render and 'num_gaussians_perpixel' in sub_df.columns
    g_val = sub_df['num_gaussians_perpixel'].mean() if has_g else None
    
    # Initialize rows
    row1 = ["", ""] if has_g else [""]
    row2 = ["", "#G / im"] if has_g else [""]
    row3 = [exp_name, fmt('num_gaussians_perpixel', g_val)] if has_g else [exp_name]
    
    # Append "Average" block
    row1.extend(["Average"] + [""] * (len(metrics) - 1))
    row2.extend([m.replace('train_', '') for m in metrics])
    row3.extend([fmt(m, avg_row[m]) for m in metrics])
    
    # Append blocks for each category
    for cat in categories:
        row1.extend([str(cat)] + [""] * (len(metrics) - 1))
        row2.extend([m.replace('train_', '') for m in metrics])
        cat_row = df_group.loc[cat]
        row3.extend([fmt(m, cat_row[m]) for m in metrics])
        
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--model", type=str, default='offthegrid')
    parser.add_argument("--ckpt_folder", type=str, default='./checkpoints/offthegrid/')
    parser.add_argument("--output_dir", type=str, default='./results/evaluation_0/')
    parser.add_argument("--align_mode", type=str, default='posefree_noalign')
    parser.add_argument("--density_mode", type=str, default='adaptive_entropy_image')
    parser.add_argument("--dataset", type=str, default='DL3DV')
    parser.add_argument("--render_train_images", type=str2bool, default=True)
    parser.add_argument("--render_test_images", type=str2bool, default=True)
    parser.add_argument("--loading_mode", type=str, default='center_crop')
    parser.add_argument("--load_size", type=int, default=518)
    parser.add_argument("--target_size", type=int, default=518)
    parser.add_argument("--save_ply", type=str2bool, default=True)
    parser.add_argument("--shared_intrinsics", type=str2bool, default=False)
    parser.add_argument("--render_videos", type=str2bool, default=True)
    parser.add_argument("--evaluate_rendering", type=str2bool, default=True)
    parser.add_argument("--test_time_optimization", type=str2bool, default=False)
    parser.add_argument("--save_camera_params", type=str2bool, default=False)
    parser.add_argument("--num_iterations_tto", type=int, default=10)
    parser.add_argument("--test_camera_pose_optimization", type=str2bool, default=False)
    parser.add_argument("--evaluate_depth", type=str2bool, default=False)
    parser.add_argument("--evaluate_rendered_depth", type=str2bool, default=False)
    parser.add_argument("--save_depth", type=str2bool, default=False)
    parser.add_argument("--evaluate_camera_pose", type=str2bool, default=False)
    parser.add_argument("--generate_json", type=str2bool, default=False)
    parser.add_argument("--encoder_model", type=str, default='vggt')
    parser.add_argument("--decoder_type", type=str, default='offthegrid')
    parser.add_argument("--evaluate_intrinsics", type=str2bool, default=False)
    parser.add_argument("--extrapolated_videos", type=str2bool, default=True)    
    parser.add_argument('--num_gaussians_per_patch', type=int, nargs='*', required=False)
    parser.add_argument("--scale_align", type=str2bool, default=False)
    parser.add_argument("--count_primitives", type=str2bool, default=True)    
    args = parser.parse_args()

    torch.cuda.empty_cache()

    if not args.num_gaussians_per_patch:
        args.num_gaussians_per_patch = [16,32,64]
    
    if args.model == 'offthegrid':
        model = OffTheGrid(encoder=args.encoder_model, decoder=args.decoder_type, geometry_mode='depth', num_gaussians_per_patch=args.num_gaussians_per_patch, predict_conf=True, use_features=True)
        model.test_density_mode = args.density_mode
        model.load_from_checkpoint(args.ckpt_folder)
    elif args.model == 'da3_giant':
        model = DA3GaussianModel(encoder='depthanything3_giant')

    model.training_mode = False
    
    
    if args.encoder_model != 'vggt':
        model.eval()

    if args.dataset == 'full_eval':
        
        full_eval_config = {
            'Charge': [3, 6, 9, 25],
            'SCRREAM': [3, 6, 9, 12],
            'mipnerf360': [3, 6, 9, 12],
            'TanksandTemples': [3, 6, 9, 12],
            '7scenes': [3, 6, 9, 12],
            'DL3DV': [3, 6, 9, 12]
        }
        geometry_eval_config = {
            'Charge': [3, 6, 9, 25],
            'SCRREAM': [3, 6, 9, 12],
        }
        figure_config = {
            'TanksandTemples': [3, 6, 9, 12],
            'mipnerf360': [3, 6, 9, 12],
        }
        wild_eval_config = {
            #'DL3DVsynth': [3, 6, 9, 12],
            #'RealX3D': [3, 6, 9, 12],
            #'phototourism': [3, 6, 9, 12],
            'nerfonthego': [3, 6, 9, 12],
            'RobustNeRF': [3, 6, 9, 12],
        }
        all_results = {}
        for d_name, view_configs in full_eval_config.items():
            all_results[d_name] = {}
            for v_num in view_configs:
                print(f"--- Running full_eval on {d_name} with {v_num} views ---")
                
                # Setup specific output directories
                run_output_dir = os.path.join(args.output_dir, d_name, f"{v_num}_views")
                
                avg_metrics = evaluate(args, model, d_name, v_num, run_output_dir)
                all_results[d_name][v_num] = avg_metrics
        
        # Generate requested CSV files
        save_results_tables(all_results, args.output_dir)
        print(f"Full evaluation done. Tables saved in {args.output_dir}")

    else:
        # Standard execution for a single dataset
        evaluate(args, model, args.dataset, args.num_images, args.output_dir)
        print(f"Evaluation done. Check results in {args.output_dir}")