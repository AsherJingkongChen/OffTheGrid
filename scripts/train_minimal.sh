export PYTHONPATH="`pwd`:$PYTHONPATH"

experiment_name=training_sanity_check
log_dir=./results/training_runs/
num_epochs=200
mixed_precision=bf16

#Training Data
training_datasets='minimal_dataset'
source_min_range=1
source_max_range=2
images_per_gpu=2
num_target=0
sample_mode=linear_close
loading_mode=no_crop
load_size=518

# Student model
pretrained_ckpt=None #./checkpoints/offthegrid/
encoder_model=vggt
decoder_type=offthegrid
num_gaussians_per_patch='16 32 64'
shared_intrinsics=True

# Teacher model
teacher_mode=static
teacher_ckpt=./checkpoints/vggt/

python train.py --training_datasets $training_datasets  --log_dir $log_dir/$experiment_name/ --sample_mode $sample_mode --image_min_range $source_min_range --image_max_range $source_max_range\
 --num_target $num_target  --num_epochs $num_epochs --teacher_mode $teacher_mode --pretrained_ckpt $pretrained_ckpt --images_per_gpu $images_per_gpu --shared_intrinsics $shared_intrinsics \
 --decoder_type $decoder_type --teacher_ckpt $teacher_ckpt --loading_mode $loading_mode --load_size $load_size --mixed_precision $mixed_precision --encoder_model $encoder_model \
  --num_gaussians_per_patch $num_gaussians_per_patch