#!/bin/bash

ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)


ALPHA1S=(0.7 0.8 0.9 1.0)
ALPHA2S=(0.3 0.2 0.1 0.0)


for i in "${!ALPHA1S[@]}"; do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    # bert-base-uncased_wiki_2018_finetune_moco
    for year in {2018..2021}; do
        python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/post_situatedQA/build_evaluate_retrieved_passages_dataset.py \
        --test_data_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/post_situatedQA/yearly/test/${year}/temp.test.jsonl \
        --retrieved_passages_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_test_${year}.jsonl \
        --output_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_retriever_evaluate_${year}.jsonl
    done
done

for i in "${!ALPHA1S[@]}"; do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    # bert-base-uncased_wiki_2018_finetune_moco
    for year in {2018..2021}; do
        echo "interp_${ALPHA1}_${ALPHA2}_${year}"
        python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/contriever/evaluate_retrieved_passages.py \
        --data /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_retriever_evaluate_${year}.jsonl \
        --output_log_file /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_retriever_evaluate_${year}.log
    done
done



ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5)
retrieve_top_k=10
output_retrieved_date=False
is_add_question_date=True

# bert-base-uncased
for i in "${!ALPHA1S[@]}"; do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    # Single bert-base-uncased
    for y in 2018 2021; do
        qa_path="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/post_situatedQA/yearly/test/${y}/temp.test.jsonl"
        retrieved_data_path=/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_test_${y}.jsonl
        out_file_path="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/Mixtral_SituatedQA_test_${y}.jsonl"
        batch_size=5
        if [ -f $out_file_path ]; then
            echo "File $out_file_path exists"
        else
            CUDA_VISIBLE_DEVICES=0,1 python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/generation/mixtral_8x7b.py \
                --qa_path $qa_path \
                --retrieved_data_path $retrieved_data_path \
                --retrieve_top_k $retrieve_top_k \
                --is_add_question_date \
                --out_file_path $out_file_path \
    qu            --batch_size $batch_size \
                --shard_weight
        fi
    done
done