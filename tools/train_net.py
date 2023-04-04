import glob
import os
import random
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from dataset.base_dataset import BaseDataset, build_dataloader
from transformers import get_constant_schedule_with_warmup
from open_flamingo import create_model_and_transforms
from tools.parser import read_args,random_seed
from tools.train.distributed import world_info_from_env, init_distributed_device

def main():
    args = read_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args) # TODO multi-GPU training.
    random_seed(args.seed)

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path, # "ViT-L-14"
        args.vision_encoder_pretrained, # "openai"
        args.lm_path, # 'facebook/opt-125m'
        args.tokenizer_path if args.tokenizer_path else args.lm_path, # 'facebook/opt-125m'
        cross_attn_every_n_layers=args.cross_attn_every_n_layers, # 1
        use_local_files=args.offline, # False
        use_media_placement_augmentation=args.use_media_placement_augmentation, # True
    )
    tokenizer.padding_side = "right"

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    device_id = args.rank % torch.cuda.device_count()
    model = model.to(device_id)

    # for debug
    ddp_model = model

    ############# DATASET #############
    dataset_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    dataset_cfg.Dataset.Img_Features_File_Map = dataset_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    dataset_cfg.Dataset.Object_Features_File_Map = dataset_cfg.Dataset.Object_Features_File_Map[args.obj_feats]

    train_dataset = BaseDataset(
        config=dataset_cfg.Dataset,
        split=args.split
    )
    dataset, dataloader, sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        dist=False,
        training=True
    )

    ############# TEST #############
    pbar = tqdm(dataloader,desc="iterate dataset: ")
    for i, data_dict in enumerate(pbar):
        text = tokenizer(
            data_dict['input_text'],
            max_length=64,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        input_ids, attention_mask = text["input_ids"], text["attention_mask"]

    # TODO training:
    # ddp_model = DDP(model, device_ids=[device_id])

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)

    # total_training_steps = (
    #   train_num_samples // (args.batch_size * args.world_size)
    # ) * args.num_epochs
    # if args.rank == 0:
    #     print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        NotImplementedError
    elif args.lr_scheduler == "cosine":
        NotImplementedError
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # TODO : check if a checkpoint exists for this run

    # TODO : resume from checkpoint
    resume_from_epoch = 0

    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        NotImplementedError


if __name__ == "__main__":
    main()