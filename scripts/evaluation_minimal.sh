export PYTHONPATH="`pwd`:$PYTHONPATH"
device=0
dataset=minimal_dataset
load_mode=vggt_crop
load_size=518
output_dir=./results/evaluation_minimal/
num_images=6 #not used here
#model options
model=offthegrid
encoder_model=vggt


decoder_type=offthegrid
ckpt_folder=./checkpoints/offthegrid/
num_gaussians_per_patch='16 32 64'
density_mode=adaptive_entropy_image
shared_intrinsics=False
test_time_optimization=False
num_iterations_tto=50

#eval options
evaluate_rendering=True
scale_align=True
evaluate_camera_pose=False
evaluate_intrinsics=False
evaluate_depth=False
evaluate_rendered_depth=False
count_primitives=True

#saving options
save_depth=True
render_train_images=True
render_test_images=True
render_videos=True
extrapolated_videos=True
save_ply=True
save_camera_params=False

CUDA_VISIBLE_DEVICES=$device python evaluation.py --dataset $dataset --model $model --output_dir $output_dir \
        --ckpt_folder $ckpt_folder --num_images $num_images --render_train_images $render_train_images \
        --render_test_images $render_test_images --render_videos $render_videos --evaluate_rendering $evaluate_rendering\
        --extrapolated_videos $extrapolated_videos --save_ply $save_ply \
        --test_time_optimization $test_time_optimization --num_iterations_tto $num_iterations_tto \
        --shared_intrinsics $shared_intrinsics --evaluate_intrinsics $evaluate_intrinsics --evaluate_depth $evaluate_depth \
        --save_camera_params $save_camera_params --evaluate_camera_pose $evaluate_camera_pose --evaluate_rendered_depth $evaluate_rendered_depth \
        --save_depth $save_depth --loading_mode $load_mode --load_size $load_size --num_gaussians_per_patch $num_gaussians_per_patch \
        --encoder_model $encoder_model --decoder_type $decoder_type --scale_align $scale_align --count_primitives $count_primitives --density_mode $density_mode
