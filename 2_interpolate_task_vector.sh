#!/bin/bash

ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

# Add time vectors together with each alpha value and evaluate on all years
for i in "${!ALPHA1S[@]}"; do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/contriever/2_contriever_interpolate_vector.py \
        --path_to_source_model facebook/contriever \
        --task_vectors "/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/bert-base-uncased_wikipedia_2018_train_moco_2018_rankloss_1_queue_131072_lr_2e-6_beir_1/task_vector/" \
        "/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/bert-base-uncased_wikipedia_2021_train_moco_2021_rankloss_1_queue_131072_lr_2e-6_beir_1/task_vector/" \
        --lambdas $ALPHA1 $ALPHA2 \
        --output_dir "/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}"
done