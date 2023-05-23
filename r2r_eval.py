import os
import pickle
import random
import re
from tensorboardX import SummaryWriter
import numpy as np
from copy import deepcopy
from PIL import Image
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
# from dataset.base_dataset import BaseDataset, build_dataloader
from dataset.environment import R2RDataset,build_dataloader
from transformers import get_constant_schedule_with_warmup
from open_flamingo import create_model_and_transforms
from tools.parser import read_args,random_seed
from tools.train.distributed import world_info_from_env, init_distributed_device
from tools.train.train_utils import (
    get_grouped_params, check_checkpoint,
    get_checkpoint, save_checkpoint,
)
from tools.finetune_utils import (
    evaluate
)
import datetime


def get_tokenizer_token_ids(tokenizer):
    image_tokens = ['<image{}>'.format(x) for x in range(12)]
    action_tokens = ['<walkto{}>'.format(_) for _ in range(12)] + ['<stop>']
    image_token_ids = tokenizer.encode(
        "".join(image_tokens),add_special_tokens=False
    )
    action_token_ids = tokenizer.encode(
        "".join(action_tokens),add_special_tokens=False
    )
    return image_token_ids, action_token_ids


def main():
    args = read_args()
    # validation: w/o dataset shuffle
    args.r2r_tok = True  # add new special tokens to tokenizer.
    args.shuffle = False

    ############# CONFIGURATION #############
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    global_cfg.Dataset.Img_Features_File_Map = global_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    global_cfg.Dataset.Object_Features_File_Map = global_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    args.enable_imgdataset = False if global_cfg.Dataset.get('IMG_DIR',None) is None else True
    args.max_length = global_cfg.Dataset.tokenizer.max_length

    device_id = init_distributed_device(args)  # TODO multi-GPU training.

    log_file = Path(args.run_name) / ('eval_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    from tools import common_utils
    logger = common_utils.create_logger(log_file, rank=args.rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_config_to_file(global_cfg, logger=logger)

    random_seed(args.seed)

    ############# MODEL #############
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path, # "ViT-L-14"
        args.vision_encoder_pretrained, # "openai"
        args.lm_path, # 'facebook/opt-125m'
        args.tokenizer_path if args.tokenizer_path else args.lm_path, # 'facebook/opt-125m'
        enable_offline_vision_encoder=not args.enable_imgdataset,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers, # 1
        use_local_files=args.offline, # False
        use_media_placement_augmentation=args.use_media_placement_augmentation, # True
        unfreeze_llm=args.unfreeze_llm, # unfreeze language model
        args=args,
    )

    ################### Word Tokens ###################
    # <PAD> on the left
    tokenizer.padding_side = "left"
    args.image_token_ids, args.action_token_ids = get_tokenizer_token_ids(tokenizer)
    args.endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    logger.info("**************************** Eval ****************************")

    ############# DATASET #############
    r2r_dataset = R2RDataset(
        config=global_cfg.Dataset,
        training=False if args.split != 'train' else True,
        logger=logger,
        args=args,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    r2r_dataset, r2r_dataloader, r2r_sampler = build_dataloader(
        dataset=r2r_dataset, batch_size=args.batch_size, distributed=args.distributed, workers=args.workers, training=False
    )

    ############# Init #############
    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")
    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus
    model = model.to(device_id)
    optimizer = torch.optim.AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps
    )

    resume_from_epoch = 0
    # TODO : check if a checkpoint exists for this run
    resume_from_epoch, global_step = check_checkpoint(args, model, optimizer, lr_scheduler, logger)

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[device_id])
        # args.batch_size: BATCH_SIZE_PER_GPU
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        total_gpus = 1
        ddp_model = model
        logger.info('Training with a single process')

    total_training_steps = (
                            len(r2r_dataset) // (args.batch_size * args.world_size)
                        ) * args.num_epochs

    logger.info(f"Total training steps: {total_training_steps}")
    tb_log = SummaryWriter(log_dir=str(Path(args.run_name) / 'tensorboard')) if args.rank == 0 else None

    ############# EVAL #############

    evaluate(
        args=args,
        model=model,
        r2r_dataset=r2r_dataset,
        r2r_dataloader=r2r_dataloader,
        tokenizer=tokenizer,
        device_id=device_id,
        logger=logger
    )

    logger.info("******************** END ********************")




if __name__ == "__main__":
    main()