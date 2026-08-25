export PYTHONPATH="`pwd`:$PYTHONPATH"
device=0
dataset=full_eval
load_mode=vggt_crop
load_size=518
output_dir=./results/evaluation_paper/
num_images=12 #not used here
#model options
model=offthegrid
encoder_model=vggt


decoder_type=offthegrid
ckpt_folder=./checkpoints/offthegrid/
num_gaussians_per_patch='16 32 64'
density_mode=adaptive_entropy_image
shared_intrinsics=False

#eval options
evaluate_rendering=True
scale_align=True
evaluate_camera_pose=True
evaluate_intrinsics=True
evaluate_depth=True
evaluate_rendered_depth=True
count_primitives=True

#saving options
save_depth=True
render_train_images=True
render_test_images=True
render_videos=False
extrapolated_videos=True
save_ply=False
save_camera_params=False

CUDA_VISIBLE_DEVICES=$device python evaluation.py --dataset $dataset --model $model --output_dir $output_dir \
        --ckpt_folder $ckpt_folder --num_images $num_images --render_train_images $render_train_images \
        --render_test_images $render_test_images --render_videos $render_videos --evaluate_rendering $evaluate_rendering\
        --extrapolated_videos $extrapolated_videos --save_ply $save_ply \
        --shared_intrinsics $shared_intrinsics --evaluate_intrinsics $evaluate_intrinsics --evaluate_depth $evaluate_depth \
        --save_camera_params $save_camera_params --evaluate_camera_pose $evaluate_camera_pose --evaluate_rendered_depth $evaluate_rendered_depth \
        --save_depth $save_depth --loading_mode $load_mode --load_size $load_size --num_gaussians_per_patch $num_gaussians_per_patch \
        --encoder_model $encoder_model --decoder_type $decoder_type --scale_align $scale_align --count_primitives $count_primitives --density_mode $density_mode
