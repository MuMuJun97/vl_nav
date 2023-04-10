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
                max_length=128,
                padding="longest",
                truncation=True, # "only_first"?train
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
                answer_labels[bs, st_idx:ed_idx] = -100

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





