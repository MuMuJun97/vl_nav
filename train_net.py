import json
import os
import pickle
import time
import numpy as np
from copy import deepcopy
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from dataset.base_dataset import BaseDataset, build_dataloader
from transformers import get_constant_schedule_with_warmup
from open_flamingo import create_model_and_transforms
from tools.parser import read_args,random_seed
from tools.validation_utils import (
    val_one_epoch,train_one_epoch
)
from tools.train.distributed import world_info_from_env, init_distributed_device
from tools.train.train_utils import (
    get_grouped_params, check_checkpoint,
    get_checkpoint, save_checkpoint,
)
import datetime
from tensorboardX import SummaryWriter

def main():
    args = read_args()

    ############# CONFIGURATION #############
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    global_cfg.Dataset.Img_Features_File_Map = global_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    global_cfg.Dataset.Object_Features_File_Map = global_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    args.enable_imgdataset = False if global_cfg.Dataset.get('IMG_DIR',None) is None else True
    args.max_length = global_cfg.Dataset.tokenizer.max_length

    device_id = init_distributed_device(args) # TODO multi-GPU training.

    log_file = Path(args.run_name) / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

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
    )

    ################### Word Tokens ###################
    # <PAD> on the left
    tokenizer.padding_side = "left"
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    ### "<image> .Question:"
    question_token_id = tokenizer(".Question", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("?Answer", add_special_tokens=False)["input_ids"][-1]

    args.media_token_id = media_token_id
    args.endofchunk_token_id = endofchunk_token_id
    args.question_token_id = question_token_id
    args.answer_token_id = answer_token_id

    if args.split == 'train':
        logger.info("**************************** Training ****************************")

        ############# DATASET #############
        dataset = BaseDataset(
            config=global_cfg.Dataset,
            split=args.split,
            training=False if args.split != 'train' else True,
            rank=args.rank
        )
        dataset, dataloader, sampler = build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            dist=args.distributed,
            workers=args.workers,
            training=False if args.split != 'train' else True
        )

        if args.trainval_step > 0:
            val_dataset = BaseDataset(
                config=global_cfg.Dataset,
                split="val_unseen",
                training=False
            )
            val_dataset, val_dataloader, val_sampler = build_dataloader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                dist=args.distributed,
                workers=args.workers,
                training=False
            )

        random_seed(args.seed, args.rank)

        print(f"Start running training on rank {args.rank}.")

        # args.rank: global rank.
        total_gpus = torch.cuda.device_count()
        device_id = args.rank % total_gpus
        model = model.to(device_id)

        optimizer = torch.optim.AdamW(get_grouped_params(model, args), lr=args.learning_rate)
        if args.lr_scheduler == "linear":
            raise NotImplementedError
        elif args.lr_scheduler == "cosine":
            raise NotImplementedError
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

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
            len(dataset) // (args.batch_size * args.world_size)
        ) * args.num_epochs

        logger.info(f"Total training steps: {total_training_steps}")
        tb_log = SummaryWriter(log_dir=str(Path(args.run_name) / 'tensorboard')) if args.rank == 0 else None

        ############# MODEL-TRAIN #############
        ddp_model.train()
        min_val_loss = 1e+10
        for epoch in range(resume_from_epoch, args.num_epochs):
            global_step = train_one_epoch(
                args=args,
                model=ddp_model,
                epoch=epoch,
                data_loader=dataloader,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device_id=device_id,
                tb_log=tb_log,
                logger=logger
            )

            ### train with validation
            if args.trainval_step > 0 and epoch % args.trainval_step == 0:
                val_loss,val_pred_dict = val_one_epoch(
                    args=args,
                    model=ddp_model,
                    epoch=epoch,
                    data_loader=val_dataloader,
                    tokenizer=tokenizer,
                    global_cfg=global_cfg,
                    device_id=device_id,
                    tb_log=tb_log,
                    logger=logger
                )
                # val_pred_file = Path(args.run_name) / (
                #             'val_pred_{}-{}.json'.format(
                #                 datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
                #                 epoch
                #             ))
                # with open(str(val_pred_file),'w') as f:
                #     json.dump(val_pred_dict,f,indent=2)

                logger.info("[Training with Validation Loss {:.2f}]".format(val_loss))
            else:
                val_loss = 0

            if args.rank == 0 and epoch % args.save_ckpt_step == 0:
                min_val_loss = val_loss
                save_checkpoint(args, epoch, ddp_model, optimizer, lr_scheduler, logger, global_step, min_val_loss)

        if args.rank == 0:
            # save final weights
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)
            torch.save(get_checkpoint(ddp_model), f"{args.run_name}/final_weights.pt")
    else:
        print("[ERROR] Please use \"eval_net.py\" for evaluation")
        exit()

    logger.info("************************ END ************************")

if __name__ == "__main__":
    main()