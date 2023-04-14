import glob
import os
import sys
from pathlib import Path
pro = Path(__file__).parent.parent.resolve()
sys.path.append(str(pro))
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
from tools.parser import read_args, random_seed
from tools.train.distributed import world_info_from_env, init_distributed_device


def get_grouped_params(model, args):
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


def check_checkpoint(args, ddp_model, optimizer, lr_scheduler):
    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )
    # TODO : resume from checkpoint
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1
    return resume_from_epoch


def train_one_epoch(
        args, model, epoch, data_loader, tokenizer, optimizer, lr_scheduler, device_id
):
    raise NotImplementedError


def main():
    args = read_args()

    device_id = init_distributed_device(args)  # TODO multi-GPU training.

    random_seed(args.seed)

    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,  # "ViT-L-14"
        args.vision_encoder_pretrained,  # "openai"
        args.lm_path,  # 'facebook/opt-125m'
        args.tokenizer_path if args.tokenizer_path else args.lm_path,  # 'facebook/opt-125m'
        enable_offline_vision_encoder=True,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,  # 1
        use_local_files=args.offline,  # False
        use_media_placement_augmentation=args.use_media_placement_augmentation,  # True
    )
    tokenizer.padding_side = "left"

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus
    model = model.to(device_id)

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[device_id])
        # args.batch_size: BATCH_SIZE_PER_GPU
        print('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        total_gpus = 1
        ddp_model = model
        print('Training with a single process')

    ############# DATASET #############
    dataset_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    dataset_cfg.Dataset.Img_Features_File_Map = dataset_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    dataset_cfg.Dataset.Object_Features_File_Map = dataset_cfg.Dataset.Object_Features_File_Map[args.obj_feats]

    train_dataset = BaseDataset(
        config=dataset_cfg.Dataset,
        split=args.split
    )
    train_dataset, train_dataloader, sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        dist=args.distributed,
        training=True
    )

    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model, args), lr=args.learning_rate)

    total_training_steps = (
                                   len(train_dataset) // (args.batch_size * args.world_size)
                           ) * args.num_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        raise NotImplementedError
    elif args.lr_scheduler == "cosine":
        raise NotImplementedError
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # TODO : check if a checkpoint exists for this run
    resume_from_epoch = check_checkpoint(args, ddp_model, optimizer, lr_scheduler)

    ddp_model.train()

    # TEST DATASET
    ############# TEST #############
    pbar = tqdm(train_dataloader,desc="iterate dataset: ")
    for i, data_dict in enumerate(pbar):
        text = tokenizer(
            data_dict['input_text'],
            max_length=64,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        input_ids, attention_mask = text["input_ids"], text["attention_mask"]
        print(input_ids.shape)

if __name__ == "__main__":
    main()