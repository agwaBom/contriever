# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import random
import json
import sys
import numpy as np
from src import normalize_text
from tqdm import tqdm
from .data import randomcrop, apply_augmentation, add_bos_eos, build_mask

class HardDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.5,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            positive = random.choice(example["positive_ctxs"])
            weak_positive = random.choice(example["weak_positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            hard_negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"], n_hard_negatives
                )
                hard_negatives += hard_negatives
        else:
            positive = example["positive_ctxs"][0]
            weak_positive = example["weak_positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []

            if "hard_negative_ctxs" in example:
                hard_negatives = [example["hard_negative_ctxs"][nidx]]
            else:
                hard_negatives = []


        example = {
            "query": self.normalize_fn(question),
            "positive": self.normalize_fn(positive),
            "weak_positive": self.normalize_fn(weak_positive),
            "negatives": [self.normalize_fn(n) for n in negatives],
            "hard_negatives": [self.normalize_fn(n) for n in hard_negatives],
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in tqdm(fin):
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives

class HardCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        poss = [ex["positive"] for ex in batch]
        weak_poss = [ex["weak_positive"] for ex in batch]

        negs = [item for ex in batch for item in ex["negatives"]]
        hardnegs = [item for ex in batch for item in ex["hard_negatives"]]
        allpassages = poss + weak_poss + negs + hardnegs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        p_tokens, p_mask = k_tokens[:len(poss)], k_mask[:len(poss)]
        wp_tokens, wp_mask = k_tokens[len(poss):len(poss)+len(weak_poss)], k_mask[len(poss):len(poss)+len(weak_poss)]
        n_tokens, n_mask = k_tokens[len(poss)+len(weak_poss):len(poss)+len(weak_poss)+len(negs)], k_mask[len(poss)+len(weak_poss)+len(negs):]
        hn_tokens, hn_mask = k_tokens[len(poss)+len(weak_poss)+len(negs):], k_mask[len(poss)+len(weak_poss)+len(negs):]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "p_tokens": p_tokens,
            "p_mask": p_mask,
            "wp_tokens": wp_tokens,
            "wp_mask": wp_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
            "hn_tokens": hn_tokens,
            "hn_mask": hn_mask,
        }
        return batch

class PositiveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.5,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]

        if self.training:
            # question = self._random_space_crop(example["question"])
            # positive = self._random_space_crop(random.choice(example["positive_ctxs"]))
            # weak_positive = self._random_space_crop(random.choice(example["weak_positive_ctxs"]))
            # weak_positive = self._random_space_crop(example["weak_positive_ctxs"][0])

            # no crop
            question = example["question"]
            positive = random.choice(example["positive_ctxs"])
            weak_positive = random.choice(example["weak_positive_ctxs"])
        else:
            question = example["question"]
            positive = example["positive_ctxs"][0]
            weak_positive = example["weak_positive_ctxs"][0]

        example = {
            "query": self.normalize_fn(question),
            "positive": self.normalize_fn(positive),
            "weak_positive": self.normalize_fn(weak_positive),
        }
        return example
    
    def _random_space_crop(self, text, ratio_min=0.1, ratio_max=1.0):
        # Trim the string to ensure we're only dealing with leading and trailing spaces
        trimmed_str = text.split()
        #trimmed_str = text

        # Calculate the cropping ratio
        ratio = random.uniform(ratio_min, ratio_max)

        length = int(len(trimmed_str) * ratio)

        start = random.randint(0, len(trimmed_str) - length)
        end = start + length
        crop = ' '.join(trimmed_str[start:end])
        #crop = trimmed_str[start:end]
        return crop

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in tqdm(fin):
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter



class PositiveCollator(object):
    def __init__(self, tokenizer, chunk_length, opt, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.chunk_length = chunk_length
        self.passage_maxlength = passage_maxlength
        self.opt = opt

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        poss = [ex["positive"] for ex in batch]
        weak_poss = [ex["weak_positive"] for ex in batch]

        allpassages = poss + weak_poss

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        p_tokens, p_mask = k_tokens[:len(poss)], k_mask[:len(poss)]
        wp_tokens, wp_mask = k_tokens[len(poss):len(poss)+len(weak_poss)], k_mask[len(poss):len(poss)+len(weak_poss)]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "p_tokens": p_tokens,
            "p_mask": p_mask,
            "wp_tokens": wp_tokens,
            "wp_mask": wp_mask,
        }
        return batch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
    ):
        self.negative_ctxs = negative_ctxs
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            gold = random.choice(example["positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []
        
        #gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"]
        #negatives = [n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives]

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        return batch
