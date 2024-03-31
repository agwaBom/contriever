#!/bin/bash

for year in {2012..2014}; do
        CUDA_VISIBLE_DEVICES=0 python finetuning.py \
                --retriever_model_id bert-base-uncased --pooling average \
                --train_data encoded-data/bert-base-uncased/wmt_yearly_data/splitted/${year}_train --loading_mode split \
                --eval_data encoded-data/bert-base-uncased/wmt_yearly_data/splitted/${year}_dev \
                --model_path facebook/contriever \
                --chunk_length 256 \
                --momentum 0.9995 --temperature 0.05 \
                --total_steps 20000 --lr 0.00001 \
                --scheduler linear --optim adamw --per_gpu_batch_size 1024 \
                --output_dir ./checkpoint/contriever_wmt_${year}_finetune_inbatch \
                --eval_freq 1000 \
                --save_freq 1000 \
                --contrastive_mode inbatch
done