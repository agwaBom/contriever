import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_task_vector(pretrained_model, finetuned_model, alpha=None):
    res = deepcopy(pretrained_model)
    task_vec = _get_task_vector(pretrained_model, finetuned_model, alpha=alpha)
    res.load_state_dict(task_vec)
    #candidate_ft = task_op(pretrained_model, res, op='add', alpha=alpha)
    # assert is_same_model(candidate_ft, finetuned_model)
    return res

def _get_task_vector(pretrained_model, finetuned_model, alpha: float = 1.0):
    pretrained_sd = pretrained_model.state_dict()
    finetuned_sd = finetuned_model.state_dict()
    with torch.no_grad():
        merged = {}
        for key in finetuned_sd:
            if pretrained_sd[key].shape != finetuned_sd[key].shape:
                print(f"Shape mismatch for {key}: {pretrained_sd[key].shape} vs {finetuned_sd[key].shape}")
                import IPython; IPython.embed(); exit()

            if pretrained_sd[key].dtype == torch.int64 or pretrained_sd[key].dtype == torch.int32:
                "dtype int64 or int32"
                import IPython; IPython.embed(); exit()
            merged[key] = alpha * (finetuned_sd[key].to("cpu") - pretrained_sd[key].to("cpu"))
    return merged
            

if __name__ == "__main__":
    import argparse
    import os
    import src.contriever
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="facebook/contriever")
    parser.add_argument("--finetuned_model_path", type=str, default="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/processed/checkpoint/add_title/contriever_wikipedia_2018_train_moco_2018_rankloss_1_queue_131072_lr_2e-6_beir_1/checkpoint/step-500000/")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="task_vector")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pre_model, tokenizer, _ = src.contriever.load_retriever(args.pretrained_model_path)
    print('Pretrained model')
    print(torch.sum(pre_model.encoder.layer[11].output.dense.weight))

    fine_model, _, _ = src.contriever.load_retriever(args.finetuned_model_path)
    print('Fine-tuned model')
    print(torch.sum(fine_model.encoder.layer[11].output.dense.weight))

    time_vec = get_task_vector(pre_model, fine_model, alpha=args.alpha)
    print('Time vector')
    print(torch.sum(time_vec.encoder.layer[11].output.dense.weight))   

    time_vec.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)