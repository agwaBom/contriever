#!/bin/bash

for year in {2018..2021}; do
    python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/post_situatedQA/build_evaluate_retrieved_passages_dataset.py \
    --test_data_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/post_situatedQA/yearly/test/${year}/temp.test.jsonl \
    --retrieved_passages_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/things_to_send/run_situatedqa/situatedqa_indexes/20171220-20181220/contriever/qa_2018_retrieved_result_top_10.jsonl \
    --output_path /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/things_to_send/run_situatedqa/situatedqa_indexes/20171220-20181220/contriever/qa_2018_retrieved_evaluate_top_10.jsonl
done

for year in {2018..2021}; do
    echo "interp_${ALPHA1}_${ALPHA2}_${year}"
    python /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/src/retriever/contriever/contriever/evaluate_retrieved_passages.py \
    --data /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_retriever_evaluate_${year}.jsonl \
    --output_log_file /home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_${ALPHA1}_${ALPHA2}/SituatedQA_retriever_evaluate_${year}.log
done
