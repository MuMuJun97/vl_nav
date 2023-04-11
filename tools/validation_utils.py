import copy
from copy import deepcopy
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from tools.train.train_utils import get_autocast, get_cast_dtype, AverageMeter
from dataset.base_dataset import BaseDataset, build_dataloader


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


def text_generate(args, global_cfg, model, tokenizer, device_id):
    """
    @param args:
    @param global_cfg:
    @param model:
    @param tokenizer:
    @param device_id: 0
    """
    args.split = 'train'
    args.generate_nums = 40
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
                desc="validation {}:".format(args.split)
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

            batch_pred = dict()
            for bs in range(outputs.shape[0]):
                sample_idx = batch_dict['sample_idx'][bs]
                batch_pred[sample_idx] = dict()
                batch_pred[sample_idx]['input_text'] = batch_dict['input_text'][bs]
                batch_pred[sample_idx]['pred_text'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[bs]
                batch_pred[sample_idx]['input_answer_text'] = input_answer_text[bs]

            predictions.update(batch_pred)


            ############## VALIDATION LOSS ##############
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
            loss_metric.accumulate(loss.data.item())

    val_loss = loss_metric.average
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
        predictions[sample_idx]['pred_text'] = tokenizer.batch_decode(outputs, skip_special_tokens=True)[bs]
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