import copy
from copy import deepcopy
import os
import time
import numpy as np
from pathlib import Path
import datetime
import torch
import json
from tqdm import tqdm
from tools.train.train_utils import get_autocast, get_cast_dtype, AverageMeter
from dataset.base_dataset import BaseDataset, build_dataloader
from dataset.generation_dataset import MP3DGenerationDataset
from tools.common_utils import all_gather
from tools.parser import random_seed

class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        return self.total / self.num


def validation(args, global_cfg, model, tokenizer, device_id):
    """
    @param args:
    @param global_cfg:
    @param model:
    @param tokenizer:
    @param device_id: 0
    """
    dataset = BaseDataset(
        config=global_cfg.Dataset,
        split=args.split,
        training=False if args.split != 'train' else True
    )
    dataset, dataloader, sampler = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        dist=args.distributed,
        workers=args.workers,
        training=False if args.split != 'train' else True
    )

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    question_token_id = tokenizer("Question", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("?Answer", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.eval()
    predictions = []
    loss_metric = Metrics()

    with torch.no_grad():
        for idx, batch_dict in tqdm(
                enumerate(dataloader),
                total=dataloader.num_batches,
                disable=args.rank != 0,
                desc="validation {}:".format(args.split)
        ):
            #### FORWARD PASS ####
            if batch_dict.get('imgs', None) is not None:
                # (B, T_img=12, F, C, H, W) with F=1
                #  Batch_size, T_img: num_media=12, F: num_frames
                input_imgs = batch_dict['imgs'].to(device_id, dtype=cast_dtype, non_blocking=True)
                use_local_vision = 'none'
                input_imgs = input_imgs.unsqueeze(2)
            elif batch_dict.get('img_feats', None) is not None:
                input_imgs = batch_dict['img_feats'].to(device_id, dtype=cast_dtype, non_blocking=True)
                use_local_vision = 'feature'
            else:
                raise NotImplementedError

            input_text = tokenizer(
                batch_dict['input_text'],
                max_length=args.max_length,
                padding="longest",
                truncation=True,  # "only_first"?train
                return_tensors="pt",
            )
            input_ids = input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = input_text['attention_mask'].to(
                device_id, dtype=cast_dtype, non_blocking=True
            )

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, 0] = -100
            labels[labels == media_token_id] = -100

            # question->answer:
            answer_labels = labels.clone()
            for bs in range(labels.shape[0]):
                st_idx = (answer_labels[bs] == question_token_id).nonzero(as_tuple=True)[0]
                ed_idx = (answer_labels[bs] == answer_token_id).nonzero(as_tuple=True)[0]
                ed_idx += 2 # "?Answer:"
                answer_labels[bs, :ed_idx] = -100

            # labels.to(device_id)
            answer_labels.to(device_id)

            # TODO: Text Generate
            # with torch.inference_mode():
            #     outputs = model.generate(
            #         input_imgs.to(device_id if device_id >= 0 else "cpu"),
            #         input_ids.to(device_id if device_id >= 0 else "cpu"),
            #         attention_mask=attention_mask.to(device_id if device_id >= 0 else "cpu"),
            #         max_new_tokens=global_cfg.Inference.max_new_tokens,
            #         num_beams=global_cfg.Inference.num_beams,
            #         length_penalty=global_cfg.Inference.length_penalty,
            #     )
            # outputs = outputs[:, len(input_ids[0]):]

            loss = model(
                vision_x=input_imgs,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=answer_labels,
                use_local_vision=use_local_vision,
            )[0]
            loss_metric.accumulate(loss.data.item())


def text_generate(args, global_cfg, model, tokenizer, device_id, logger):
    """
    @param args:
    @param global_cfg:
    @param model:
    @param tokenizer:
    @param device_id: 0
    @param logger: log message
    """
    logger.info("*************** generate text | {} split ***************".format(args.generate_split))

    # eval: if split == 'train', set training=True to keep the same train-dataset settings.
    dataset = BaseDataset(
        config=global_cfg.Dataset,
        split=args.generate_split,
        training=False if args.generate_split != 'train' else True,
        generate_start_index=args.generate_start_index
    )
    dataset, dataloader, sampler = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        dist=args.distributed,
        workers=args.workers,
        training=False if args.generate_split != 'train' else True,
    )

    # Keep the seed the same as it was during training: train_net.py
    random_seed(args.seed, args.rank)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.eval()
    predictions = dict()
    loss_metric = Metrics()
    pred_num = 0

    with torch.no_grad():
        for idx, batch_dict in tqdm(
                enumerate(dataloader),
                total=dataloader.num_batches,
                disable=args.rank != 0,
                desc="validation {}:".format(args.generate_split)
        ):
            pred_num += 1
            if pred_num > args.generate_nums:
                break

            #### FORWARD PASS ####
            if batch_dict.get('imgs', None) is not None:
                # (B, T_img=12, F, C, H, W) with F=1
                #  Batch_size, T_img: num_media=12, F: num_frames
                input_imgs = batch_dict['imgs'].to(device_id, dtype=cast_dtype, non_blocking=True)
                use_local_vision = 'none'
                input_imgs = input_imgs.unsqueeze(2)
            elif batch_dict.get('img_feats', None) is not None:
                input_imgs = batch_dict['img_feats'].to(device_id, dtype=cast_dtype, non_blocking=True)
                use_local_vision = 'feature'
            else:
                raise NotImplementedError

            #### TEXT GENERATION | ANSWER MASK ####
            input_question_text = deepcopy(batch_dict['input_text'])
            input_answer_text = deepcopy(batch_dict['input_text'])
            for bs in range(len(input_question_text)):
                st_idx = input_question_text[bs].find('Answer:')
                input_question_text[bs] = input_question_text[bs][:st_idx] + "Answer:"
                input_answer_text[bs] = input_answer_text[bs][st_idx:].replace("Answer:", "")

            input_question_text = tokenizer(
                input_question_text,
                max_length=args.max_length,
                padding="longest",
                truncation=True,  # "only_first"?train
                return_tensors="pt",
            )
            input_ids = input_question_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = input_question_text['attention_mask'].to(
                device_id, dtype=cast_dtype, non_blocking=True
            )

            ############### greedy inference | text generation ###############
            # Greedy Inference
            generate_outputs = model(
                vision_x=input_imgs,
                lang_x=input_ids,
                attention_mask=attention_mask,
                mode='generate',
                max_length=global_cfg.Inference.max_new_tokens,
            )

            # Single-GPU Inference
            # [1] global_cfg.Inference.num_beams=1 greedy generation
            # outputs = model.generate(
            #     input_imgs.to(device_id if device_id >= 0 else "cpu"),
            #     input_ids.to(device_id if device_id >= 0 else "cpu"),
            #     attention_mask=attention_mask.to(device_id if device_id >= 0 else "cpu"),
            #     max_new_tokens=global_cfg.Inference.max_new_tokens,
            #     num_beams=global_cfg.Inference.num_beams,
            #     length_penalty=global_cfg.Inference.length_penalty,
            # )

            generate_outputs = generate_outputs[:, len(input_ids[0]):]

            ############## VALIDATION LOSS ##############
            input_text = tokenizer(
                batch_dict['input_text'],
                max_length=args.max_length,
                padding="longest",
                truncation="only_first", # True,  # "only_first"?train
                return_tensors="pt",
            )
            input_ids = input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = input_text['attention_mask'].to(
                device_id, dtype=cast_dtype, non_blocking=True
            )
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, 0] = -100
            labels[labels == args.media_token_id] = -100

            # question->answer:
            answer_labels = labels.clone()
            for bs in range(labels.shape[0]):
                # LLaMa: tokenizer.decode([0,1,2,32000,32001,32002])
                #   >>> '<unk><s></s><|endofchunk|><image><PAD>'
                # st_idx = (answer_labels[bs] == args.question_token_id).nonzero(as_tuple=True)[0]
                ed_idx = (answer_labels[bs] == args.answer_token_id).nonzero(as_tuple=True)[0]
                ed_idx += 2  # "?Answer:"
                answer_labels[bs, :ed_idx] = -100

            # labels.to(device_id)
            answer_labels.to(device_id)
            ####### Forward, Compute Loss #######
            with autocast():
                output = model(
                    vision_x=input_imgs,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=answer_labels,
                    use_local_vision=use_local_vision,
                )
                loss = output[0]
                logits = output[1]
            loss_metric.accumulate(loss.data.item())

            ####### Loss -> Text Generation #######
            batch_pred = train_with_generate(
                args, batch_dict, tokenizer, logits, loss, answer_labels, generate_outputs
            )
            all_batch_pred = all_gather(batch_pred)
            if args.rank == 0:
                for per_pred in all_batch_pred:
                    predictions.update(per_pred)

    val_loss = loss_metric.average
    predictions['val_mean_loss'] = val_loss
    return val_loss,predictions

def inference_text_generation(
        args,
        model,
        batch_dict,
        tokenizer,
        global_cfg,
        device_id,
        cast_dtype
):
    #### FORWARD PASS ####
    if batch_dict.get('imgs', None) is not None:
        # (B, T_img=12, F, C, H, W) with F=1
        #  Batch_size, T_img: num_media=12, F: num_frames
        input_imgs = batch_dict['imgs'].to(device_id, dtype=cast_dtype, non_blocking=True)
        use_local_vision = 'none'
        input_imgs = input_imgs.unsqueeze(2)
    elif batch_dict.get('img_feats', None) is not None:
        input_imgs = batch_dict['img_feats'].to(device_id, dtype=cast_dtype, non_blocking=True)
        use_local_vision = 'feature'
    else:
        raise NotImplementedError

    #### TEXT GENERATION | ANSWER MASK ####
    input_question_text = deepcopy(batch_dict['input_text'])
    input_answer_text = deepcopy(batch_dict['input_text'])
    for bs in range(len(input_question_text)):
        st_idx = input_question_text[bs].find('Answer:')
        input_question_text[bs] = input_question_text[bs][:st_idx] + "Answer:"
        input_answer_text[bs] = input_answer_text[bs][st_idx:].replace("Answer:","")

    input_text = tokenizer(
        input_question_text,
        max_length=args.max_length,
        padding="longest",
        truncation=True,  # "only_first"?train
        return_tensors="pt",
    )
    input_ids = input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
    attention_mask = input_text['attention_mask'].to(
        device_id, dtype=cast_dtype, non_blocking=True
    )

    with torch.inference_mode():
        outputs = model.generate(
            input_imgs.to(device_id if device_id >= 0 else "cpu"),
            input_ids.to(device_id if device_id >= 0 else "cpu"),
            attention_mask=attention_mask.to(device_id if device_id >= 0 else "cpu"),
            max_new_tokens=global_cfg.Inference.max_new_tokens,
            num_beams=global_cfg.Inference.num_beams,
            length_penalty=global_cfg.Inference.length_penalty,
        )
    outputs = outputs[:, len(input_ids[0]):]

    predictions = dict()
    for bs in range(outputs.shape[0]):
        sample_idx = batch_dict['sample_idx'][bs]
        predictions[sample_idx] = dict()
        predictions[sample_idx]['input_text'] = batch_dict['input_text'][bs]
        predictions[sample_idx]['pred_text'] = tokenizer.batch_decode(outputs, skip_special_tokens=False)[bs]
        predictions[sample_idx]['input_answer_text'] = input_answer_text[bs]

    return predictions


def forward_with_loss(
        args,
        model,
        batch_dict,
        tokenizer,
        device_id,
        cast_dtype
):
    #### FORWARD PASS ####
    if batch_dict.get('imgs', None) is not None:
        # (B, T_img=12, F, C, H, W) with F=1
        #  Batch_size, T_img: num_media=12, F: num_frames
        input_imgs = batch_dict['imgs'].to(device_id, dtype=cast_dtype, non_blocking=True)
        use_local_vision = 'none'
        input_imgs = input_imgs.unsqueeze(2)
    elif batch_dict.get('img_feats', None) is not None:
        input_imgs = batch_dict['img_feats'].to(device_id, dtype=cast_dtype, non_blocking=True)
        use_local_vision = 'feature'
    else:
        raise NotImplementedError

    ########### Compute Val Loss ###########
    input_text = tokenizer(
        batch_dict['input_text'],
        max_length=args.max_length,
        padding="longest",
        truncation=True,  # "only_first"?train
        return_tensors="pt",
    )
    input_ids = input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
    attention_mask = input_text['attention_mask'].to(
        device_id, dtype=cast_dtype, non_blocking=True
    )

    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[:, 0] = -100
    labels[labels == args.media_token_id] = -100

    # question->answer:
    answer_labels = labels.clone()
    for bs in range(labels.shape[0]):
        st_idx = (answer_labels[bs] == args.question_token_id).nonzero(as_tuple=True)[0]
        ed_idx = (answer_labels[bs] == args.answer_token_id).nonzero(as_tuple=True)[0]
        ed_idx += 2  # "?Answer:"
        answer_labels[bs, :ed_idx] = -100

    # labels.to(device_id)
    answer_labels.to(device_id)

    loss = model(
        vision_x=input_imgs,
        lang_x=input_ids,
        attention_mask=attention_mask,
        labels=answer_labels,
        use_local_vision=use_local_vision,
    )[0]

    return loss


# def train_one_epoch_debug(
#         args,model,epoch,data_loader,tokenizer,optimizer,lr_scheduler,device_id,tb_log=None,logger=None
# ):
#     num_batches_per_epoch = data_loader.num_batches
#     total_training_steps = num_batches_per_epoch * args.num_epochs
#     data_loader = range(len(data_loader))
#     for num_steps, batch_dict in tqdm(
#             enumerate(data_loader),
#             disable=args.rank != 0,
#             total=total_training_steps,
#             initial=(epoch * num_batches_per_epoch)
#     ):
#         global_step = num_steps + epoch * num_batches_per_epoch
#         tb_log.add_scalar('train/loss', global_step, global_step)
#     return global_step

def train_with_generate(args,batch_dict,tokenizer,logits,loss,answer_labels,generate_outputs=None):
    """
    Args:
        args:
        batch_dict:
        tokenizer:
        logits: predictions during training.
        loss:
    Note:
        :code: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ... # input [<s> Token Token...] -> pred [NextToken NextToken ...]
                # labels [<s> GT GT ...]
                # loss = CrossEntropyLoss( logits[..., :-1, :], labels[..., 1:] )
            loss = loss_fct(shift_logits, shift_labels)
    Returns:

    """
    batch_pred = dict()
    with torch.no_grad():
        input_question_text = deepcopy(batch_dict['input_text'])
        input_answer_text = deepcopy(batch_dict['input_text'])
        for bs in range(len(input_question_text)):
            st_idx = input_question_text[bs].find('Answer:')
            input_question_text[bs] = input_question_text[bs][:st_idx] + "Answer:"
            input_answer_text[bs] = input_answer_text[bs][st_idx:].replace("Answer:", "")

            answer_sdx = (answer_labels[bs] != -100).nonzero(as_tuple=True)[0][0]
            answer_sdx = answer_sdx-1 # Shift so that tokens < n predict n logits

            answer_logits = logits[bs, answer_sdx:-1]
            pred_tokens = torch.argmax(answer_logits, dim=-1)

            sample_idx = batch_dict['sample_idx'][bs]
            batch_pred[sample_idx] = dict()
            batch_pred[sample_idx]['_gt_full_text'] = batch_dict['input_text'][bs]

            ### input: full length sequence (include question + answer)
            batch_pred[sample_idx]['pred_answer_text'] = \
                tokenizer.decode(pred_tokens, skip_special_tokens=False)

            ### input: prefix generate input sequence (only question, without answer)
            if generate_outputs is not None:
                batch_pred[sample_idx]['generate_answer_text'] = \
                    tokenizer.decode(generate_outputs[bs], skip_special_tokens=False)

            question_text = input_question_text[bs][input_question_text[bs].find("Question"):].replace(
                "Answer:", ""
            )
            batch_pred[sample_idx]['_gt_question_text'] = question_text
            batch_pred[sample_idx]['_gt_answer_text'] = input_answer_text[bs]
            batch_pred[sample_idx]['mean_loss'] = loss.data.item()

    return batch_pred

def train_one_epoch(
    args,model,epoch,data_loader,tokenizer,optimizer,lr_scheduler,device_id,tb_log=None,logger=None
):
    model.train()
    loss_metric = Metrics()

    num_batches_per_epoch = data_loader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of Data (= 1 batch regardless of gradient accum)
    end = time.time()
    predictions = dict() if args.rank == 0 else None

    for num_steps, batch_dict in tqdm(
        enumerate(data_loader),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(epoch*num_batches_per_epoch)
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### FORWARD PASS ####
        if batch_dict.get('imgs',None) is not None:
            # (B, T_img=12, F, C, H, W) with F=1
            #  Batch_size, T_img: num_media=12, F: num_frames
            input_imgs = batch_dict['imgs'].to(device_id, dtype=cast_dtype, non_blocking=True)
            use_local_vision = 'none'
            input_imgs = input_imgs.unsqueeze(2)
        elif batch_dict.get('img_feats',None) is not None:
            input_imgs = batch_dict['img_feats'].to(device_id, dtype=cast_dtype, non_blocking=True)
            use_local_vision = 'feature'
        else:
            raise NotImplementedError

        input_text = tokenizer(
            batch_dict['input_text'],
            max_length=args.max_length,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        input_ids = input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = input_text['attention_mask'].to(
            device_id, dtype=cast_dtype, non_blocking=True
        )

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100 # <PAD> LLaMa
        labels[:, 0] = -100 # first token <s> or <PAD>
        labels[labels == args.media_token_id] = -100

        # question->answer:
        answer_labels = labels.clone()
        for bs in range(labels.shape[0]):
            # LLaMa: tokenizer.decode([0,1,2,32000,32001,32002])
            #   >>> '<unk><s></s><|endofchunk|><image><PAD>'
            # st_idx = (answer_labels[bs] == args.question_token_id).nonzero(as_tuple=True)[0]
            ed_idx = (answer_labels[bs] == args.answer_token_id).nonzero(as_tuple=True)[0]
            ed_idx += 2  # "?Answer:"
            answer_labels[bs,:ed_idx] = -100

        # labels.to(device_id)
        answer_labels.to(device_id)

        with autocast():
            output = model(
                vision_x=input_imgs,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=answer_labels,
                use_local_vision=use_local_vision,
            )
            loss = output[0]
            logits = output[1]
        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[args.media_token_id] = torch.ones_like(zero_mask[args.media_token_id])
                zero_mask[args.endofchunk_token_id] = torch.ones_like(
                    zero_mask[args.endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if args.train_with_generate:
            batch_pred = train_with_generate(
                args,batch_dict,tokenizer,logits,loss,answer_labels
            )
            all_batch_pred = all_gather(batch_pred)
            if args.rank == 0:
                for per_pred in all_batch_pred:
                    predictions.update(per_pred)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_metric.accumulate(loss.data.item())

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if tb_log is not None:
                try:
                    cur_lr = float(optimizer.lr)
                except:
                    cur_lr = optimizer.param_groups[0]['lr']

                tb_log.add_scalar('meta_data/learning_rate',cur_lr,global_step)
                tb_log.add_scalar('train/loss', loss.data.item(), global_step)

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            logger.info(
                f"\nStep {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. "
                f"\nAverage Loss: {loss_metric.average:.3f}"
            )

    if args.train_with_generate and args.rank == 0:
        train_pred_file = Path(args.run_name) / (
                    'train_{}_pred_{}.json'.format(
                        epoch,
                        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                    ))
        with open(str(train_pred_file),'w') as f:
            json.dump(predictions,f,indent=2)

    return global_step


def val_one_epoch(args,model,epoch,data_loader,tokenizer,global_cfg,device_id,tb_log=None,logger=None):
    model.eval()

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    loss_metric = Metrics()
    predictions = dict()

    with torch.no_grad():
        for idx, batch_dict in tqdm(
                enumerate(data_loader),
                total=data_loader.num_batches,
                disable=args.rank != 0,
                desc="validation {}:".format(args.split)
        ):
            loss = forward_with_loss(
                args=args,
                model=model,
                batch_dict=batch_dict,
                tokenizer=tokenizer,
                device_id=device_id,
                cast_dtype=cast_dtype
            )
            loss_metric.accumulate(loss.data.item())

            # cur_preds = inference_text_generation(
            #     args=args,
            #     model=model,
            #     batch_dict=batch_dict,
            #     tokenizer=tokenizer,
            #     global_cfg=global_cfg,
            #     device_id=device_id,
            #     cast_dtype=cast_dtype
            # )
            # predictions.update(cur_preds)

    val_loss = loss_metric.average
    return val_loss,None


def mp3d_text_generation(args, global_cfg, model, tokenizer, device_id, logger):
    """
    @param args:
    @param global_cfg:
    @param model:
    @param tokenizer:
    @param device_id: 0
    @param logger: log message
    """
    logger.info("*************** generate language instruction from Matterport3D ***************")

    # eval: if split == 'train', set training=True to keep the same train-dataset settings.
    dataset = MP3DGenerationDataset(
        config=global_cfg.Dataset,
        split=args.generate_split,
        training=False if args.generate_split != 'train' else True,
        generate_start_index=args.generate_start_index
    )
    dataset, dataloader, sampler = build_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        dist=args.distributed,
        workers=args.workers,
        training=False if args.generate_split != 'train' else True,
    )

    # Keep the seed the same as it was during training: train_net.py
    random_seed(args.seed, args.rank)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.eval()
    predictions = dict()
    pred_num = 0

    with torch.no_grad():
        for idx, batch_dict in tqdm(
                enumerate(dataloader),
                total=dataloader.num_batches,
                disable=args.rank != 0,
                desc="validation {}:".format(args.generate_split)
        ):
            pred_num += 1
            if pred_num > args.generate_nums:
                break

            #### FORWARD PASS ####
            if batch_dict.get('imgs', None) is not None:
                # (B, T_img=12, F, C, H, W) with F=1
                #  Batch_size, T_img: num_media=12, F: num_frames
                input_imgs = batch_dict['imgs'].to(device_id, dtype=cast_dtype, non_blocking=True)
                use_local_vision = 'none'
                input_imgs = input_imgs.unsqueeze(2)
            elif batch_dict.get('img_feats', None) is not None:
                input_imgs = batch_dict['img_feats'].to(device_id, dtype=cast_dtype, non_blocking=True)
                use_local_vision = 'feature'
            else:
                raise NotImplementedError

            #### TEXT GENERATION | ANSWER MASK ####
            input_question_text = deepcopy(batch_dict['input_text'])
            input_answer_text = deepcopy(batch_dict['input_text'])
            for bs in range(len(input_question_text)):
                st_idx = input_question_text[bs].find('Answer:')
                input_question_text[bs] = input_question_text[bs][:st_idx] + "Answer:"
                input_answer_text[bs] = input_answer_text[bs][st_idx:].replace("Answer:", "")

            input_question_text = tokenizer(
                input_question_text,
                max_length=args.max_length,
                padding="longest",
                truncation=True,  # "only_first"?train
                return_tensors="pt",
            )
            input_ids = input_question_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = input_question_text['attention_mask'].to(
                device_id, dtype=cast_dtype, non_blocking=True
            )

            ############### greedy inference | text generation ###############
            # Greedy Inference
            generate_outputs = model(
                vision_x=input_imgs,
                lang_x=input_ids,
                attention_mask=attention_mask,
                mode='generate',
                max_length=global_cfg.Inference.max_new_tokens,
            )

            generate_outputs = generate_outputs[:, len(input_ids[0]):]

            batch_pred = dict()
            input_question_text = deepcopy(batch_dict['input_text'])
            input_answer_text = deepcopy(batch_dict['input_text'])
            for bs in range(len(input_question_text)):
                st_idx = input_question_text[bs].find('Answer:')
                input_question_text[bs] = input_question_text[bs][:st_idx] + "Answer:"
                input_answer_text[bs] = input_answer_text[bs][st_idx:].replace("Answer:", "")

                sample_idx = batch_dict['sample_idx'][bs]
                batch_pred[sample_idx] = dict()
                batch_pred[sample_idx]['_gt_full_text'] = batch_dict['input_text'][bs]

                ### input: prefix generate input sequence (only question, without answer)
                batch_pred[sample_idx]['generate_answer_text'] = \
                    tokenizer.decode(generate_outputs[bs], skip_special_tokens=False)

                question_text = input_question_text[bs][input_question_text[bs].find("Question"):].replace(
                    "Answer:", ""
                )
                batch_pred[sample_idx]['_gt_question_text'] = question_text

            all_batch_pred = all_gather(batch_pred)
            if args.rank == 0:
                for per_pred in all_batch_pred:
                    predictions.update(per_pred)

    return predictions
