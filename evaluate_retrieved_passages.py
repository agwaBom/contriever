# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import glob

import numpy as np
import torch

import src.utils

from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)
# input_data=[{'answer': answer, 'ctxs': [{'text': text}, {'text': text2}]}, ... ,{}]
def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    #logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    #logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits


def main(opt):
    logger = src.utils.init_logger(opt, stdout_only=True)
    datapaths = glob.glob(args.data)
    r10, r100 = [], []
    for path in datapaths:
        data = []
        with open(path, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
            #data = json.load(fin)
        answers = [ex['answers'] for ex in data]
        top_k_hits = validate(data, args.validation_workers)
        message = f"Evaluate results from {path}:"
        for k in [1, 5, 10, 20, 100]:
            if k <= len(top_k_hits):
                recall = 100 * top_k_hits[k-1]
                if k == 10:
                    r10.append(f"{recall:.1f}")
                if k == 100:
                    r100.append(f"{recall:.1f}")
                message += f' R@{k}: {recall:.1f}'
        logger.info(message)
    print(datapaths)
    print('\t'.join(r10))
    print('\t'.join(r100))
    with open(opt.output_log_file, 'a') as f:
        f.write(message.replace(f"Evaluate results from {path}: ", "")+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/time_vector/bert-base-uncased/timemoco_bert-base-uncased_2018_2021_interp_0.3_0.7/SituatedQA_retriever_evaluate_2018.jsonl")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--output_log_file', type=str, default='./tmp.log',)

    args = parser.parse_args()
    main(args)
