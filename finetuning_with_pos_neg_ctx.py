# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
import time
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils, contriever, finetuning_data, inbatch, moco
import gc
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def finetuning(opt, model, optimizer, scheduler, tokenizer, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    if hasattr(model, "module"):
        eval_model = model.module
    else:
        eval_model = model
    eval_model = eval_model.get_encoder()

    print("Loading training data (Est 3 min - 1,500,000it)")
    train_dataset = finetuning_data.PositiveDataset(#HardDataset(
        datapaths=opt.train_data,
        negative_ctxs=opt.negative_ctxs,
        negative_hard_ratio=opt.negative_hard_ratio,
        negative_hard_min_idx=opt.negative_hard_min_idx,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=True,
        # maxload=50000, #debugging
    )

    print("Loading evaluation data (Est 30 sec - 180,000it)")
    if len(opt.eval_data) != 0:
        dev_dataset = finetuning_data.PositiveDataset(# HardDataset(
            datapaths=opt.eval_data,
            negative_ctxs=opt.negative_ctxs,
            negative_hard_ratio=opt.negative_hard_ratio,
            negative_hard_min_idx=opt.negative_hard_min_idx,
            normalize=opt.eval_normalize_text,
            global_rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            maxload=50000,#int(opt.maxload * 0.2) if opt.maxload is not None else 50000,
            training=False,
        )

    #collator = finetuning_data.HardCollator(tokenizer, passage_maxlength=opt.chunk_length)
    collator = finetuning_data.PositiveCollator(tokenizer, chunk_length=opt.chunk_length, opt=opt, passage_maxlength=opt.chunk_length)
    #train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        #sampler=train_sampler,
        shuffle=False,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        #num_workers=opt.num_workers,
        collate_fn=collator,
    )
    if len(opt.eval_data) != 0:
        #dev_sampler = SequentialSampler(dev_dataset)
        dev_dataloader = DataLoader(
            dev_dataset,
            #sampler=dev_sampler,
            shuffle=False,
            batch_size=opt.per_gpu_batch_size,
            drop_last=False,
            #num_workers=opt.num_workers,
            collate_fn=collator,
        )

    # 여기는 확인이 필요해보임
    # train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
    # evaluate(opt, eval_model, tokenizer, tb_logger, step)

    epoch = 1
    kill_step = 500000
    
    model.eval()
    encoder = model.get_encoder()
    evaluate_model(
        opt, query_encoder=encoder, doc_encoder=encoder, tokenizer=tokenizer, tb_logger=tb_logger, step=step, alias="q_p"
    )
    if opt.contrastive_mode == "moco3":
        encoder = model.get_encoder(return_encoder_q_wp=True)
        evaluate_model(
            opt, query_encoder=encoder, doc_encoder=encoder, tokenizer=tokenizer, tb_logger=tb_logger, step=step, alias="q_wp"
        )
    
    model.train()
    prev_ids, prev_mask = None, None
    while step < opt.total_steps and step < kill_step:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        for i, batch in enumerate(train_dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            step += 1
            # train_loss, iter_stats = model(**batch, stats_prefix="train")
            # train_loss.backward()
            # tokenizer.batch_decode(batch["q_tokens"], skip_special_tokens=True)
            #import IPython; IPython.embed(); exit(1)
            train_loss, iter_stats = model(**batch, stats_prefix="train")
            train_loss.backward(retain_graph=True)

            if opt.optim == "sam" or opt.optim == "asam":
                optimizer.first_step(zero_grad=True)

                sam_loss, _ = model(**batch, stats_prefix="train/sam_opt")
                sam_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    if "train" in k:
                        log += f" | {k}: {v:.3f}"
                        if tb_logger:
                            tb_logger.add_scalar(k, v, step)
                tb_logger.add_scalar("learning rate", scheduler.get_last_lr()[0], step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()
            
            if opt.eval_freq != 0 and step % opt.eval_freq == 0:
                gc.collect()
                encoder = model.get_encoder()
                evaluate_model(
                    opt, query_encoder=encoder, doc_encoder=encoder, tokenizer=tokenizer, tb_logger=tb_logger, step=step, alias="q_p"
                )
                if opt.contrastive_mode == "moco3":
                    encoder = model.get_encoder(return_encoder_q_wp=True)
                    evaluate_model(
                        opt, query_encoder=encoder, doc_encoder=encoder, tokenizer=tokenizer, tb_logger=tb_logger, step=step, alias="q_wp"
                    )
                # dev set validation
                # validate(model, dev_dataloader, tb_logger, step)

                # train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
                # evaluate(opt, eval_model, tokenizer, tb_logger, step)

                if step % opt.save_freq == 0 and dist_utils.get_rank() == 0:
                    utils.save(
                        model,
                        optimizer,
                        scheduler,
                        step,
                        opt,
                        opt.output_dir,
                        f"step-{step}",
                    )
                model.train()

            if step >= opt.total_steps or step >= kill_step:
                break

            #if step % 20 == 0:
                #gc.collect() # slows down training
            #    summary.print_(summary.summarize(muppy.get_objects()))

        epoch += 1
        
        '''
        model.p_queue = torch.randn(opt.projection_size, opt.queue_size).cuda()
        model.p_queue = torch.nn.functional.normalize(model.p_queue, dim=0)
        model.p_queue_ptr = torch.zeros(1, dtype=torch.long).cuda()

        model.wp_queue = torch.randn(opt.projection_size, opt.queue_size).cuda()
        model.wp_queue = torch.nn.functional.normalize(model.wp_queue, dim=0)
        model.wp_queue_ptr = torch.zeros(1, dtype=torch.long).cuda()
        '''

def evaluate_model(opt, query_encoder, doc_encoder, tokenizer, tb_logger, step, alias):
    datasetnamelist=["scifact", "concatenated_2018", "concatenated_2019", "concatenated_2020", "concatenated_2021"]

    for datasetname in datasetnamelist:
        if datasetname == "scifact":
            metrics = beir_utils.evaluate_model(
                query_encoder,
                doc_encoder,
                tokenizer,
                dataset=datasetname,
                batch_size=opt.per_gpu_batch_size,
                norm_doc=opt.norm_doc,
                norm_query=opt.norm_query,
                beir_dir=opt.eval_datasets_dir,
                score_function=opt.score_function,
                lower_case=opt.lower_case,
                normalize_text=opt.eval_normalize_text,
            )
            message = []
            if dist_utils.is_main():
                for metric in ["NDCG@10", "Recall@10", "Recall@100"]:
                    message.append(f"{datasetname}/{alias}/{metric}: {metrics[metric]:.2f}")
                    if tb_logger is not None:
                        tb_logger.add_scalar(f"{datasetname}/{alias}/{metric}", metrics[metric], step)
                logger.info(" | ".join(message))
        else:
            metrics = beir_utils.evaluate_model(
                query_encoder,
                doc_encoder,
                tokenizer,
                dataset=datasetname,
                batch_size=opt.per_gpu_batch_size,
                norm_doc=opt.norm_doc,
                norm_query=opt.norm_query,
                beir_dir="/home/work/khyunjin1993/dev/myrepo/temporal_alignment_rag/dataset/wikidpr_dataset/contriever_finetuning_data/original/situated_qa_beir_1003/",
                score_function=opt.score_function,
                lower_case=opt.lower_case,
                normalize_text=opt.eval_normalize_text,
            )
            message = []
            if dist_utils.is_main():
                for metric in ["NDCG@10", "Recall@10", "Recall@100", "2018", "2019", "2020", "2021"]:
                    message.append(f"{datasetname}/{alias}/{metric}: {metrics[metric]:.2f}")
                    if tb_logger is not None:
                        tb_logger.add_scalar(f"{datasetname}/{alias}/{metric}", metrics[metric], step)
                logger.info(" | ".join(message))
    
    


def validate(model, dev_dataloader, tb_logger, step):
    model.eval()
    total_dev_p_loss = 0
    total_dev_wp_loss = 0
    total_dev_rank_loss = 0
    total_dev_all_loss = 0
    total_dev_acc = 0
    total_dev_stdq = 0
    total_dev_stdp = 0
    total_dev_stdwp = 0
    with torch.no_grad():
        for dev_batch in dev_dataloader:
            dev_batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in dev_batch.items()}
            dev_loss, iter_stats_dev = model(**dev_batch, stats_prefix="dev")

            total_dev_p_loss += iter_stats_dev["dev/p_loss"][0]
            total_dev_wp_loss += iter_stats_dev["dev/wp_loss"][0]
            total_dev_rank_loss += iter_stats_dev["dev/rank_loss"][0]
            total_dev_all_loss += iter_stats_dev["dev/all_loss"][0]
            total_dev_acc += iter_stats_dev["dev/accuracy"][0]
            total_dev_stdq += iter_stats_dev["dev/stdq"][0]
            total_dev_stdp += iter_stats_dev["dev/stdp"][0]
            total_dev_stdwp += iter_stats_dev["dev/stdwp"][0]

    tb_logger.add_scalar("dev/accuracy", total_dev_acc/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/p_loss", total_dev_p_loss/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/wp_loss", total_dev_wp_loss/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/rank_loss", total_dev_rank_loss/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/all_loss", total_dev_all_loss/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/stdq", total_dev_stdq/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/stdp", total_dev_stdp/len(dev_dataloader), step)
    tb_logger.add_scalar("dev/stdwp", total_dev_stdwp/len(dev_dataloader), step)


def evaluate(opt, model, tokenizer, tb_logger, step):
    dataset = finetuning_data.Dataset(
        datapaths=opt.eval_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=False,
    )
    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    model.eval()
    if hasattr(model, "module"):
        model = model.module
    correct_samples, total_samples, total_step = 0, 0, 0
    all_q, all_g, all_n = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
            all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

            q_emb = model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=opt.norm_query)
            all_emb = model(input_ids=all_tokens, attention_mask=all_mask, normalize=opt.norm_doc)

            g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

            all_q.append(q_emb)
            all_g.append(g_emb)
            all_n.append(n_emb)

        all_q = torch.cat(all_q, dim=0)
        all_g = torch.cat(all_g, dim=0)
        all_n = torch.cat(all_n, dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        message = []
        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            logger.info(" | ".join(message))
            if tb_logger is not None:
                tb_logger.add_scalar(f"eval_acc", acc, step)
                tb_logger.add_scalar(f"mrr", mrr, step)


def main():
    # need to implement utils.load()
    logger.info("Start")

    options = Options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    # retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    # opt.retriever_model_id = retriever_model_id

    if opt.contrastive_mode == "moco1":
        model_class = moco.TimeMoCo
    elif opt.contrastive_mode == "moco2":
        model_class = moco.TimeMoCo2
    elif opt.contrastive_mode == "moco3":
        model_class = moco.TimeMoCo3
    elif opt.contrastive_mode == "mocoq":
        model_class = moco.TimeMoCoQ
    elif opt.contrastive_mode == "mocoq1":
        model_class = moco.TimeMoCoQ1
    elif opt.contrastive_mode == "mocoq2":
        model_class = moco.TimeMoCoQ2
    elif opt.contrastive_mode == "mocoq2_frezze_embed":
        model_class = moco.TimeMoCoQ2_Frezze_embed
    elif opt.contrastive_mode == "inbatch":
        model_class = inbatch.InBatch
    else:
        raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")

    if not directory_exists and opt.model_path == "none":
        model = model_class(opt)
        model = model.cuda()
        optimizer, scheduler = utils.set_optim(opt, model)
        step = 0
    elif not directory_exists and opt.model_path != "none":
        model, _, _, _, _ = utils.load(
            model_class,
            opt.model_path,
            opt,
            reset_params=False,
        )
        logger.info(f"2nd Phase Model loaded from {opt.output_dir}")
        step = 0
        optimizer, scheduler = utils.set_optim(opt, model)
    elif directory_exists:
        model_path = os.path.join(opt.output_dir, "checkpoint", "latest")
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            model_path,
            opt,
            reset_params=False,
        )
        logger.info(f"Model loaded from {opt.output_dir}")
    else:
        retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
        opt.retriever_model_id = retriever_model_id
 
        if opt.contrastive_mode == 'inbatch':
            model = inbatch.InBatch(opt, retriever, tokenizer)
        elif opt.contrastive_mode == 'moco':
            model = moco.TimeMoCo(opt, retriever, tokenizer)
        else:
            raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")
        model = model.cuda()
        optimizer, scheduler = utils.set_optim(opt, model)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = opt.dropout

        '''
        model, optimizer, scheduler, opt_checkpoint, step = utils.load(
            model_class,
            opt.model_path,
            opt,
            reset_params=False if opt.continue_training else True,
        )
        if not opt.continue_training:
            step = 0
        logger.info(f"Model loaded from {opt.model_path}")
        '''
    logger.info(utils.get_parameters(model))

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
        dist.barrier()

    '''
    if opt.contrastive_mode == 'inbatch':
        model = inbatch.InBatch(opt, retriever, tokenizer)
    elif opt.contrastive_mode == 'moco':
        model = moco.TimeMoCo(opt, retriever, tokenizer)
    else:
        raise ValueError(f"contrastive mode: {opt.contrastive_mode} not recognised")
    model = model.cuda()

    optimizer, scheduler = utils.set_optim(opt, model)
    # if dist_utils.is_main():
    #    utils.save(model, optimizer, scheduler, global_step, 0., opt, opt.output_dir, f"step-{0}")
    logger.info(utils.get_parameters(model))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
    '''
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)


if __name__ == "__main__":
    main()
