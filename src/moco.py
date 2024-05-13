# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import logging
import copy
import transformers

from src import contriever, dist_utils, utils
logger = logging.getLogger(__name__)

class TimeMoCo(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(TimeMoCo, self).__init__()

        self.queue_size = opt.queue_size
        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k  # apply the encoder on keys in train mode

        retriever, tokenizer = self._load_retriever(
            opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
        )

        self.tokenizer = tokenizer
        self.encoder_q = retriever
        self.encoder_k = copy.deepcopy(retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("p_queue", torch.randn(opt.projection_size, self.queue_size))
        self.p_queue = nn.functional.normalize(self.p_queue, dim=0)
        self.register_buffer("p_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("wp_queue", torch.randn(opt.projection_size, self.queue_size))
        self.wp_queue = nn.functional.normalize(self.wp_queue, dim=0)
        self.register_buffer("wp_queue_ptr", torch.zeros(1, dtype=torch.long))

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    def _momentum_update_k_encoder(self):
        """
        Update of the positive encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue_p(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())

        batch_size = keys.shape[0]

        ptr = int(self.p_queue_ptr)
        assert self.queue_size % batch_size == 0, f"{batch_size}, {self.queue_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.p_queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.p_queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_wp(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())

        batch_size = keys.shape[0]

        ptr = int(self.wp_queue_ptr)
        assert self.queue_size % batch_size == 0, f"{batch_size}, {self.queue_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.wp_queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.wp_queue_ptr[0] = ptr

    def _compute_logits_p(self, q, k):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.p_queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        return logits

    def _compute_logits_wp(self, q, k):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.wp_queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        return logits

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):
        bsz = q_tokens.size(0)

        # Query encoder를 통해 Query의 feature를 추출
        q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
 
        with torch.no_grad():  # no gradient to keys
            if stats_prefix == "train":
                # update the key encoder
                self._momentum_update_k_encoder()  # update the key encoder
            elif stats_prefix == "dev" or stats_prefix == "test":
                self.encoder_k.eval()

            if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                self.encoder_k.eval()

            # 같은 K encoder를 사용하여 P(Positive), WP(Weak Positive)의 feature를 추출
            p = self.encoder_k(input_ids=kwargs['p_tokens'], attention_mask=kwargs['p_mask'], normalize=self.norm_doc)
            wp = self.encoder_k(input_ids=kwargs['wp_tokens'], attention_mask=kwargs['wp_mask'], normalize=self.norm_doc)
        
        # P와 WP의 feature를 이용하여 logits 계산 각각 다른 queue를 사용 (p_queue, wp_queue)
        p_logits = self._compute_logits_p(q, p) / self.temperature
        wp_logits = self._compute_logits_wp(q, wp) / self.temperature

        # labels: positive key indicators
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # P, WP가 정답을 맞추도록 loss 계산
        p_loss = torch.nn.functional.cross_entropy(p_logits, labels, label_smoothing=self.label_smoothing)
        wp_loss = torch.nn.functional.cross_entropy(wp_logits, labels, label_smoothing=self.label_smoothing)

        # Ranking Loss를 사용하여 WP는 P보다 항상 더 큰 loss를 가지도록 함
        rank_loss = torch.nn.functional.margin_ranking_loss(wp_loss, p_loss, target=torch.tensor(1).cuda(), margin=1.0, reduction='mean')

        # alpha값을 이용하여 P_loss와 WP_loss의 비중을 조절
        alpha = 0.7
        loss = alpha * p_loss + (1 - alpha) * wp_loss + rank_loss
        logits = alpha * p_logits + (1 - alpha) * wp_logits

        # P, WP의 feature를 queue 업데이트
        if stats_prefix == "train":
            self._dequeue_and_enqueue_p(p)
            self._dequeue_and_enqueue_wp(wp)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}p_loss"] = (p_loss.item(), bsz)
        iter_stats[f"{stats_prefix}wp_loss"] = (wp_loss.item(), bsz)
        iter_stats[f"{stats_prefix}rank_loss"] = (rank_loss.item(), bsz)
        iter_stats[f"{stats_prefix}all_loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(logits, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(q, dim=0).mean().item()
        stdp = torch.std(p, dim=0).mean().item()
        stdwp = torch.std(wp, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdp"] = (stdp, bsz)
        iter_stats[f"{stats_prefix}stdwp"] = (stdwp, bsz)

        return loss, iter_stats



class MoCo(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(MoCo, self).__init__()

        self.queue_size = opt.queue_size
        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k  # apply the encoder on keys in train mode

        retriever, tokenizer = self._load_retriever(
            opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
        )

        self.tokenizer = tokenizer
        self.encoder_q = retriever
        self.encoder_k = copy.deepcopy(retriever)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(opt.projection_size, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    def _momentum_update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f"{batch_size}, {self.queue_size}"  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def _compute_logits(self, q, k):
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        return logits

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):
        bsz = q_tokens.size(0)

        q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                self.encoder_k.eval()

            k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        
        logits = self._compute_logits(q, k) / self.temperature

        # labels: positive key indicators
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        self._dequeue_and_enqueue(k)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(logits, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(q, dim=0).mean().item()
        stdk = torch.std(k, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
