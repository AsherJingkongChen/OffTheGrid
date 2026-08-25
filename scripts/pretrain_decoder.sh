export PYTHONPATH="`pwd`:$PYTHONPATH"

experiment_name=decoder_pretrain
log_dir=/home/amoreau/vggt_checkpoints/
num_epochs=200
mixed_precision=bf16

#Training Data
training_datasets='DL3DV co3d London ARKit ScanNetplusplus Unreal re10k wildrgbd'
images_per_gpu=24
loading_mode=no_crop
load_size=518

# Student model
pretrained_ckpt=None #/home/amoreau/vggt_checkpoints/ours_finetune_rebuttal_righttemp/rebuttalv2
encoder_model=depthanything3_large
decoder_type=multires_detector



python pretrain_decoder.py --training_datasets $training_datasets  --log_dir $log_dir/$experiment_name/ \
 --num_epochs $num_epochs --pretrained_ckpt $pretrained_ckpt --images_per_gpu $images_per_gpu \
 --encoder_model $encoder_model --decoder_type $decoder_type --loading_mode $loading_mode \
 --load_size $load_size --mixed_precision $mixed_precision