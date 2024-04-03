#!/bin/bash

for year in 2012 2016 2021; do
        CUDA_VISIBLE_DEVICES=2 python finetuning.py \
                --retriever_model_id bert-base-uncased --pooling average \
                --train_data wmt_yearly_data/splitted/q_${year}_finetuning_data.jsonl \
                --loading_mode split \
                --eval_data wmt_yearly_data/splitted/q_${year}_dev_finetuning_data.jsonl \
                --model_path facebook/contriever \
                --chunk_length 256 \
                --momentum 0.9995 --temperature 0.05 \
                --warmup_steps 20000 \
                --total_steps 2000000 --lr 0.00005 \
                --scheduler linear --optim adamw --per_gpu_batch_size 32 \
                --output_dir ./checkpoint/contriever_wmt_${year}_finetune_inbatch \
                --eval_freq 5000 \
                --save_freq 5000 \
                --contrastive_mode inbatch \
                --negative_ctxs 5 \
                --negative_hard_min_idx 0 \
                --negative_hard_ratio 0.1 
done