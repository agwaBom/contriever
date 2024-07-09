#!/bin/bash

for data in nq scifact hotpotqa; do
    python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/contriever/eval_beir.py \
        --dataset ${data} \
        --beir_dir /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/beir_eval \
        --per_gpu_batch_size 64 \
        --output_dir /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/beir_eval \
        --model_name_or_path facebook/contriever \
        --norm_query \
        --norm_doc \
        --lower_case \
        --normalize_text
done



ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
path="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/original/situated_qa_beir/"
for i in "${!ALPHA1S[@]}"; do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    for data in 2018 2021; do
        CUDA_VISIBLE_DEVICES=0 python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/contriever/eval_beir.py \
            --dataset ${data} \
            --beir_dir ${path} \
            --per_gpu_batch_size 64 \
            --output_dir ${path}/${data}/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2} \
            --model_name_or_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2} \
            --norm_query \
            --norm_doc \
            --lower_case \
            --normalize_text
    done
done

ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
path="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/original/situated_qa_beir/"
for i in "${!ALPHA1S[@]}"; do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    for data in 2018 2021; do
        CUDA_VISIBLE_DEVICES=0 python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/contriever/eval_beir.py \
            --dataset ${data} \
            --beir_dir ${path} \
            --per_gpu_batch_size 64 \
            --output_dir ${path}/${data}/timemoco_contriever_2018_2021_interp_${ALPHA1}_${ALPHA2} \
            --model_name_or_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/contriever/timemoco_contriever_2018_2021_interp_${ALPHA1}_${ALPHA2} \
            --norm_query \
            --norm_doc \
            --lower_case \
            --normalize_text
    done
done