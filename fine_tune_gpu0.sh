#!/bin/bash

for year in 2012 2016 2021; do
        CUDA_VISIBLE_DEVICES=0 python finetuning_with_pos_neg_ctx.py \
                --retriever_model_id bert-base-uncased --pooling average \
                --train_data wmt_yearly_data/splitted/q_${year}_finetuning_data.jsonl \
                --loading_mode split \
                --eval_data wmt_yearly_data/splitted/q_${year}_dev_finetuning_data.jsonl \
                --model_path facebook/contriever \
                --chunk_length 256 \
                --momentum 0.9995 --temperature 0.05 \
                --warmup_steps 20000 \
                --total_steps 2000000 --lr 0.00005 \
                --scheduler linear --optim adamw --per_gpu_batch_size 2 \
                --output_dir ./checkpoint/contriever_wmt_${year}_finetune_inbatch_delete_this \
                --eval_freq 5000 \
                --save_freq 5000 \
                --contrastive_mode inbatch \
                --negative_ctxs 5 \
                --negative_hard_min_idx 0 \
                --negative_hard_ratio 0.1 
done


for year in 2012;do
        CUDA_VISIBLE_DEVICES=0 python finetuning_with_pos_neg_ctx.py \
                --retriever_model_id bert-base-uncased --pooling average \
                --train_data wmt_yearly_data_cut_tok/splitted/2012_2016_2021/q_${year}_finetuning_data.jsonl \
                --eval_data wmt_yearly_data_cut_tok/splitted/2012_2016_2021/q_${year}_dev_finetuning_data.jsonl \
                --loading_mode split \
                --augementation delete --prob_augmentation 0.1 \
                --model_path facebook/contriever \
                --chunk_length 256 \
                --ratio_min 0.1 --ratio_max 0.5 \
                --momentum 0.9995 --temperature 0.05 \
                --moco_queue 131072 \
                --warmup_steps 20000 \
                --eval_freq 1000 \
                --total_steps 2000000 --lr 0.00005 \
                --scheduler linear --optim adamw --per_gpu_batch_size 64 \
                --output_dir ./checkpoint/2012_2016_2021/contriever_wmt_${year}_finetune_moco_2012 \
                --save_freq 5000 \
                --contrastive_mode moco
done

for year in 2021; do
        CUDA_VISIBLE_DEVICES=1 python finetuning_with_pos_neg_ctx.py \
                --retriever_model_id bert-base-uncased --pooling average \
                --train_data wmt_yearly_data_cut_tok/splitted/2012_2016_2021/q_${year}_finetuning_data.jsonl \
                --eval_data wmt_yearly_data_cut_tok/splitted/2012_2016_2021/q_${year}_dev_finetuning_data.jsonl \
                --loading_mode split \
                --augementation delete --prob_augmentation 0.1 \
                --model_path facebook/contriever \
                --chunk_length 256 \
                --ratio_min 0.1 --ratio_max 0.5 \
                --momentum 0.9995 --temperature 0.05 \
                --moco_queue 131072 \
                --warmup_steps 20000 \
                --eval_freq 1000 \
                --total_steps 2000000 --lr 0.00005 \
                --scheduler linear --optim adamw --per_gpu_batch_size 64 \
                --output_dir ./checkpoint/2012_2016_2021/contriever_wmt_${year}_finetune_moco_2021 \
                --save_freq 5000 \
                --contrastive_mode moco
done