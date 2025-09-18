#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn

embed_model_path=/data/xiezequn/ICL-main/2025-CVPR-ICL/run_logs/UFine6926/20250726_171238_RDE_TAL+sr0.37_tau0.015_margin0.1_n0+aug+pre
model_dir=/data/xiezequn/ICL-main/2025-CVPR-ICL/Qwen2-VL-7B-Instruct
root_dir=/data/xiezequn/ICL-main/2025-CVPR-ICL/data

xi=0.50 # target-CUHK-PEDES/RSTPReid/UFine6926: 0.5, target-ICFG-PEDES: 0.55
lambda=0.80
source=RSTPReid  # CUHK-PEDES ICFG-PEDES RSTPReid UFine6926
target=RSTPReid # CUHK-PEDES ICFG-PEDES RSTPReid UFine6926
CUDA_VISIBLE_DEVICES=0,1,2,3  python vllm_infer_ICL.py \
        --xi ${xi} \
        --embed_model_path ${embed_model_path} \
        --source ${source} \
        --target ${target} \
        --model_dir ${model_dir} \
        --lambda ${lambda} \
        --base_model RDE \
        --root_dir ${root_dir} \
        --tag test

# conda create --name myenv python=3.10
# pip install vllm,easydict,ftfy,prettytable,nltk,qwen_vl_utils