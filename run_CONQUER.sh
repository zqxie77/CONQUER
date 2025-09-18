#!/bin/bash
root_dir=/data/xiezequn/CONQUER/data
tau=0.015 
margin=0.1
# noisy_rate=0.0  #0.0 0.2 0.5 0.8
ccl_weight=0.5  # RCL module
rematch_weight=0.5 # L2RM module
select_ratio=0.3
loss=TAL
DATASET_NAME=RSTPReid
# CUHK-PEDES ICFG-PEDES RSTPReid

# noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=2 \
    python train.py \
    --name CONQUER \
    --img_aug \
    --txt_aug \
    --batch_size 64 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}  \
    --num_epoch 60 \
    --ccl_weight $ccl_weight \
    --rematch_weight $rematch_weight