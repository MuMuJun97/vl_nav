import argparse
import random
import numpy as np
import torch
import os
from tools.train.distributed import world_info_from_env
from pathlib import Path

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def read_args():
    output_dir = Path(__file__).parent.parent.parent.resolve() / "vl_nav_output"
    parser = argparse.ArgumentParser()

    ############# DATASET #############
    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/datasets/imgdatasets.yaml", help='dataset configs')
    parser.add_argument('--img_feats', type=str, default="vit_imagenet", help='dataset configs')
    parser.add_argument('--obj_feats', type=str, default="butd_SOON", help='object features')
    parser.add_argument('--split', type=str, default="train", help='train, val, test')
    parser.add_argument('--output_dir', type=str, default=str(output_dir), help='output dir')

    ############# FLAMINGO #############
    parser.add_argument("--vision_encoder_path", default="ViT-B-16", type=str) # ViT-B-16, ViT-L-14
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    # parser.add_argument("--lm_path", default="facebook/opt-125m", type=str) # mini language model
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-125m",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=4, # LLaMa-7B 4, OPT-1.3B 1
        help="how often to add a cross-attention layer after each transformer layer",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="llmNav",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", type=bool, default=True) # False
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--logging_steps", type=int, default=2, help="log loss every n steps"
    )

    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )

    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--dataset_resampled", action="store_true")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    args = parser.parse_args()

    # local LLaMa-7B dir
    args.lm_path = args.tokenizer_path

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.run_name = Path(args.output_dir) / args.run_name
    args.run_name.mkdir(parents=True,exist_ok=True)
    args.run_name = str(args.run_name)
    return args