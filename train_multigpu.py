import os
import argparse
import torch
import itertools
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.models.off_the_grid import OffTheGrid
from src.models.geometry_encoders.vggt_encoder import VGGT_Encoder
from src.models.geometry_encoders.depthanything3_encoder import DepthAnything3_Encoder
from src.datasets.multigpu_training_dataset import create_train_dataloader
from src.datasets.vggt_data_loader import VGGTDataset
from src.datasets.Charge_dataset import ChargeDataset, charge_collate_fn
from src.datasets.scrream_dataset import SCRREAMDataset

from src.utils.os_utils import str2bool
from src.utils.loss_utils import geometry_consistency_loss, EMAHelper
from src.utils.gaussians_utils import get_cosine_schedule_with_warmup
from src.utils.eval_utils import visualize_depth

def validate(gaussian_model, accelerator, val_dataloader, iteration, tb_writer, best_psnr, args):
    gaussian_model.eval()
    all_metrics = {}
    scene_id = 0
    
    unwrapped_model = accelerator.unwrap_model(gaussian_model)

    with torch.no_grad():
        for s, val_batch in enumerate(val_dataloader):
            for key, val in val_batch.items():
                if isinstance(val, torch.Tensor):
                    val_batch[key] = val.to(accelerator.device)
            metrics = unwrapped_model.validation_step(val_batch, iteration, scene_id, tb_writer=tb_writer, tto=False)
            for key in metrics:
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key] += metrics[key]
            scene_id += val_batch["images"].shape[0]

    current_psnr = sum(all_metrics['validation_target_render/target_PSNR']) / len(all_metrics['validation_target_render/target_PSNR'])
    if accelerator.is_main_process:
        print(f"Validation Step {iteration}: Target PSNR = {current_psnr:.2f}")
    
    for key in all_metrics:
        met = sum(all_metrics[key]) / len(all_metrics[key])
        if tb_writer:
            tb_writer.add_scalar(f"{key}", met, global_step=iteration)
    
    if iteration > 1:
        save_path = os.path.join(args.log_dir, 'latest_model')
        unwrapped_model.save(save_path)
    
    if current_psnr > best_psnr:
        best_psnr = current_psnr
        save_path_best = os.path.join(args.log_dir, 'best_model')
        if iteration > 1:
            unwrapped_model.save(save_path_best)
            if accelerator.is_main_process:
                print(f"New best model saved with PSNR: {best_psnr:.2f}")

    gaussian_model.train()
    gaussian_model.training_mode = True

    return best_psnr

def main():
    parser = argparse.ArgumentParser(description="Training script parameters")
    parser.add_argument('--training_datasets', nargs='+', help='Training datasets list', required=True)
    parser.add_argument("--log_dir", type=str, default='./output/gaussians_decoder/', help="Output directory")
    parser.add_argument("--pretrained_ckpt", type=str, default='None', help="Path to pretrained checkpoint")
    parser.add_argument("--encoder_model", type=str, default='vggt', help="vggt, depthanything3")
    parser.add_argument("--decoder_type", type=str, default='multires_detector', help="Decoder architecture")
    parser.add_argument('--num_gaussians_per_patch', type=int, nargs='*', help='List that defines number of primitives per patch for each density level', required=False)
    parser.add_argument("--images_per_gpu", type=int, default=12, help="Target images per GPU")
    parser.add_argument("--num_epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--image_min_range", type=int, default=1, help="Min images per scene")
    parser.add_argument("--image_max_range", type=int, default=12, help="Max images per scene")
    parser.add_argument("--num_target", type=int, default=2, help="Max images per scene")
    parser.add_argument("--shared_intrinsics", type=str2bool, default=False, help="Average intrinsics over images")
    parser.add_argument("--sample_mode", type=str, default='linear_close', help="Sampling strategy")
    parser.add_argument("--loading_mode", type=str, default='no_crop', help="Image resizing/cropping strategy")
    parser.add_argument("--load_size", type=int, default=518, help="Image resolution (largest dimension)")
    parser.add_argument("--teacher_mode", type=str, default='static', choices=['None', 'static', 'EMA'], help="'None', 'static' or 'EMA' ")
    parser.add_argument("--ema_decay", type=float, default=0.996, help="Decay rate for EMA teacher")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision type")
    parser.add_argument("--teacher_ckpt", type=str, default='None', help="Path to teacher checkpoints folder (optional, uses HuggingFace if None)")
    parser.add_argument("--ema_update_every", type=int, default=16, help="Update teacher every N steps")
    args = parser.parse_args()


    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=args.log_dir
    )
    accelerator.even_batches = False
    set_seed(42)

    if accelerator.is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'best_model'), exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'latest_model'), exist_ok=True)
        tb_writer = SummaryWriter(args.log_dir)
        print(f"Training on {accelerator.num_processes} GPUs using {args.mixed_precision}")
    else:
        tb_writer = None

    train_dataloader = create_train_dataloader(args.training_datasets,
            num_target=args.num_target,
            image_num_range=(args.image_min_range, args.image_max_range), 
            target_mode="interpolation",
            sampling_mode='linear_close',
            loading_mode=args.loading_mode,
            load_size=args.load_size,
            max_pixels_per_gpu=args.images_per_gpu*args.load_size*args.load_size,
            num_gpus=torch.cuda.device_count(),
            num_workers=8)
    
    val_dataset = ChargeDataset(input_mode='sparse_6', num_scenes_per_seq=1, 
                                loading_mode=args.loading_mode, load_size=args.load_size)
    #val_dataset = SCRREAMDataset(num_source_views=6,
    #            num_target_views=6, 
    #            loading_mode=args.loading_mode,
    #            load_size=args.load_size,
    #            load_images=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=charge_collate_fn)

    #Student model
    gaussian_model = OffTheGrid(
        encoder=args.encoder_model, 
        decoder=args.decoder_type, 
        geometry_mode='depth',
        num_gaussians_per_patch=args.num_gaussians_per_patch, 
        predict_conf=True, 
        use_features=True
    )
    if args.pretrained_ckpt != 'None':
        gaussian_model.load_from_checkpoint(args.pretrained_ckpt)
    elif args.encoder_model == 'vggt':
        vggt_path = os.path.join(args.teacher_ckpt, '3D_foundation_model.pth') if args.teacher_ckpt != 'None' else 'None'
        if vggt_path != 'None' and os.path.exists(vggt_path):
            gaussian_model.encoder.load_state_dict(torch.load(vggt_path, map_location='cpu', weights_only=False))
        else:
            from huggingface_hub import hf_hub_download
            try:
                vggt_path_hf = hf_hub_download(repo_id='Antonion/vggt', filename='3D_foundation_model.pth')
                gaussian_model.encoder.load_state_dict(torch.load(vggt_path_hf, map_location='cpu', weights_only=False))
            except Exception as e:
                print(f"Warning: Could not find VGGT at HF repo Antonion/vggt: {e}")

    gaussian_model.training_setup(args.log_dir, accelerator.num_processes)

    #Teacher model
    teacher_model = None
    ema_helper = None

    if args.teacher_mode == 'static':
        if accelerator.is_main_process: print("Initializing Static Teacher...")
        if args.encoder_model =='vggt':
            teacher_model = VGGT_Encoder()
            vggt_path = os.path.join(args.teacher_ckpt, '3D_foundation_model.pth') if args.teacher_ckpt != 'None' else 'None'
            if vggt_path == 'None' or not os.path.exists(vggt_path):
                from huggingface_hub import hf_hub_download
                vggt_path = hf_hub_download(repo_id='Antonion/vggt', filename='3D_foundation_model.pth')
            teacher_model.load_state_dict(torch.load(vggt_path, map_location='cpu', weights_only=False))
            teacher_model.point_head = None 
        elif 'depthanything3' in args.encoder_model:
            model_type = args.encoder_model.split('_')[-1]
            if model_type == 'small':
                local_path = os.path.join(args.teacher_ckpt, "small") if args.teacher_ckpt != 'None' else 'None'
                path = local_path if local_path != 'None' and os.path.exists(local_path) else "depth-anything/DepthAnythingV3-Small"
                teacher_model = DepthAnything3_Encoder.from_pretrained(path)
            elif model_type == 'base':
                local_path = os.path.join(args.teacher_ckpt, "base") if args.teacher_ckpt != 'None' else 'None'
                path = local_path if local_path != 'None' and os.path.exists(local_path) else "depth-anything/DepthAnythingV3-Base"
                teacher_model = DepthAnything3_Encoder.from_pretrained(path)
            elif model_type == 'large':
                local_path = os.path.join(args.teacher_ckpt, "large1.1") if args.teacher_ckpt != 'None' else 'None'
                path = local_path if local_path != 'None' and os.path.exists(local_path) else "depth-anything/DepthAnythingV3-Large"
                teacher_model = DepthAnything3_Encoder.from_pretrained(path)
            elif model_type == 'giant':
                local_path = os.path.join(args.teacher_ckpt, "giant1.1") if args.teacher_ckpt != 'None' else 'None'
                path = local_path if local_path != 'None' and os.path.exists(local_path) else "depth-anything/DepthAnythingV3-Giant"
                teacher_model = DepthAnything3_Encoder.from_pretrained(path)
        teacher_model.to(accelerator.device)
        if 'depthanything3' in args.encoder_model:
            teacher_model.eval()
        teacher_model.requires_grad_(False)
    elif args.teacher_mode == 'EMA':
        if accelerator.is_main_process: print(f"Initializing EMA Teacher (decay={args.ema_decay})...")
        ema_helper = EMAHelper(gaussian_model.encoder, decay=args.ema_decay, device=accelerator.device)
        teacher_model = ema_helper.teacher 

    # --- 5. Optimizer & Scheduler ---
    if args.encoder_model =='vggt':
        encoder_lr = 1e-6
    elif 'depthanything3' in args.encoder_model:
        encoder_lr = 5e-6
    optim_params = [
        {'params': gaussian_model.encoder.parameters(), 'lr': encoder_lr},
        {'params': gaussian_model.gaussians_decoder.parameters(), 'lr': 1e-4}
    ]
    optimizer = torch.optim.AdamW(optim_params)

    num_iter_per_epoch = 1000
    total_steps = num_iter_per_epoch * args.num_epochs
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3000, num_training_steps=total_steps)

    # --- 6. Prepare with Accelerator ---
    gaussian_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        gaussian_model, optimizer, train_dataloader, scheduler
    )

    # --- 7. Training Loop ---
    if accelerator.is_main_process:
        print("Training starts")
    
    iteration = 0
    best_psnr = 0
    
    for epoch in range(args.num_epochs):
        gaussian_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not accelerator.is_main_process)
        for batch in progress_bar:
            images = batch["images"].to(accelerator.device)
            target_views = batch["target_views"].to(accelerator.device) if 'target_views' in batch else None

            teacher_data = None
            teacher_out = None

            if teacher_model is not None:
                if target_views is not None:
                    teacher_input = torch.cat([images,target_views],dim=1)
                else:
                    teacher_input = images

                with torch.no_grad():
                    teacher_out = teacher_model(teacher_input, store_embeddings=False)
                
                if target_views is not None:
                    num_targets = target_views.shape[1]
                    teacher_data = {}
                    teacher_data['images'] = target_views
                    for key in teacher_out:
                        if key != "embeddings":
                            target_data_slice = teacher_out[key][:, -num_targets:]
                            source_data_slice = teacher_out[key][:, :-num_targets]
                            
                            teacher_data[key] = target_data_slice
                            teacher_out[key] = source_data_slice 
                else:
                    teacher_data = None

            # Student Forward Pass
            gaussians, data, rendering_loss, render_log_dict = gaussian_model(images, teacher_data)
            loss = rendering_loss.mean()

            supervision_log_dict = {}
            if teacher_model is not None:                
                teacher_supervision_loss, supervision_log_dict = geometry_consistency_loss(
                    data, teacher_out, use_depth_conf=True, shared_intrinsics=args.shared_intrinsics
                )
            
                loss += teacher_supervision_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(gaussian_model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            #Prevent fragmentation build-up
            if iteration % 10 == 0:
                torch.cuda.empty_cache()
                
            # --- EMA Update ---
            if args.teacher_mode == 'EMA' and ema_helper is not None and iteration % args.ema_update_every == 0:
                student_encoder = accelerator.unwrap_model(gaussian_model).encoder
                ema_helper.update(student_encoder)

            if accelerator.is_main_process and tb_writer:
                if iteration % 20 == 0:
                    if teacher_model is not None:
                        for l in supervision_log_dict.keys():
                            tb_writer.add_scalar(l, supervision_log_dict[l], global_step=iteration)
                    for l in render_log_dict.keys():
                        tb_writer.add_scalar(l, render_log_dict[l].mean(), global_step=iteration)
                    tb_writer.add_scalar("learning_rates/lr_encoder", optimizer.param_groups[0]['lr'], global_step=iteration)
                    tb_writer.add_scalar("learning_rates/lr_gaussians_decoder", optimizer.param_groups[1]['lr'], global_step=iteration)

                if iteration % 500 == 0 and teacher_model is not None:
                    depth_visu = visualize_depth(images, data["depths"].detach().cpu(), teacher_out["depths"].detach().cpu())
                    for b in range(min(len(depth_visu), 3)):
                        tb_writer.add_images(f"depth_finetuning/depth_{b}", depth_visu[b][None], global_step=iteration)

            #Validation
            if accelerator.is_main_process:
                if iteration % num_iter_per_epoch == 0:
                    best_psnr = validate(gaussian_model, accelerator, val_dataloader, iteration, tb_writer, best_psnr, args)

            accelerator.wait_for_everyone()
            iteration += 1

    if accelerator.is_main_process:
        if tb_writer: tb_writer.close()
        print("Training complete.")

if __name__ == "__main__":
    main()