#!/bin/bash

for year in 2012 2016; do
        CUDA_VISIBLE_DEVICES=0 python train.py \
                --retriever_model_id bert-base-uncased --pooling average \
                --augmentation delete --prob_augmentation 0.1 \
                --train_data encoded-data/bert-base-uncased/wmt_yearly_data/splitted/${year}_train --loading_mode split \
                --dev_data encoded-data/bert-base-uncased/wmt_yearly_data/splitted/${year}_dev \
                --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
                --momentum 0.9995 --moco_queue 131072 --temperature 0.05 \
                --warmup_steps 20000 --total_steps 100000 --lr 0.00005 \
                --scheduler linear --optim adamw --per_gpu_batch_size 64 \
                --output_dir ./checkpoint/contriever_wmt_${year}_moco_1 \
                --eval_freq 1000 \
                --save_freq 1000 \
                --contrastive_mode moco
done

year=2021
CUDA_VISIBLE_DEVICES=1 python train.py \
        --retriever_model_id bert-base-uncased --pooling average \
        --augmentation delete --prob_augmentation 0.1 \
        --train_data encoded-data/bert-base-uncased/wmt_yearly_data/splitted/${year}_train --loading_mode split \
        --dev_data encoded-data/bert-base-uncased/wmt_yearly_data/splitted/${year}_dev \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.9995 --moco_queue 131072 --temperature 0.05 \
        --warmup_steps 20000 --total_steps 100000 --lr 0.00005 \
        --scheduler linear --optim adamw --per_gpu_batch_size 64 \
        --output_dir ./checkpoint/contriever_wmt_${year}_moco_1 \
        --eval_freq 1000 \
        --save_freq 1000 \
        --contrastive_mode moco