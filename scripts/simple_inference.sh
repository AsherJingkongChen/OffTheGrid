export PYTHONPATH="`pwd`:$PYTHONPATH"


device=0
input_dir=./data/minimal_dataset/scene_0/
output_dir=./results/simple_inference/scene_0/

num_images=6
sampling_mode=linear
loading_mode=vggt_crop
load_size=518
density_mode=adaptive_entropy_image

ckpt_folder=./checkpoints/offthegrid/
encoder_model=vggt
decoder_type=offthegrid
num_gaussians_per_patch='16 32 64'

export_to_colmap=False
save_ply=True
export_mesh=False
render_videos=True
video_trajectory=interpolation+spiral
save_geometry=True
visualize_adaptive_density=False

shared_intrinsics=False



CUDA_VISIBLE_DEVICES=$device python simple_inference.py --input_dir $input_dir --output_dir $output_dir \
    --ckpt_folder $ckpt_folder --num_images $num_images --sampling_mode $sampling_mode --render_videos $render_videos \
    --video_trajectory $video_trajectory --save_ply $save_ply --save_geometry $save_geometry \
    --shared_intrinsics $shared_intrinsics --visualize_adaptive_density $visualize_adaptive_density \
    --decoder_type $decoder_type --loading_mode $loading_mode --load_size $load_size --export_to_colmap $export_to_colmap \
    --num_gaussians_per_patch $num_gaussians_per_patch --encoder_model $encoder_model --export_mesh $export_mesh
