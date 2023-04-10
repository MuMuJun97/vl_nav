import glob
import os
import time
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
from tools.validation_utils import validation,Metrics
from tools.train.distributed import world_info_from_env, init_distributed_device
from tools.train.train_utils import get_autocast,get_cast_dtype,AverageMeter
import datetime
from tensorboardX import SummaryWriter

def get_grouped_params(model,args):
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

def check_checkpoint(args,ddp_model,optimizer,lr_scheduler,logger):
    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            logger.info(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            logger.info(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )
    # TODO : resume from checkpoint
    resume_from_epoch = global_step = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
    return resume_from_epoch, global_step

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def save_checkpoint(args, epoch, ddp_model, optimizer, lr_scheduler, logger, global_step, min_val_loss):
    if args.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        checkpoint_dict = {
            "epoch": epoch,
            "min_val_loss": min_val_loss,
            "global_step": global_step,
            "model_state_dict": get_checkpoint(ddp_model),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }
        logger.info(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch - 1}.pt")

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

def train_one_epoch(
        args,model,epoch,data_loader,tokenizer,optimizer,lr_scheduler,device_id,tb_log=None,logger=None
):
    model.train()

    num_batches_per_epoch = data_loader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    question_token_id = tokenizer("Question", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("?Answer", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of Data (= 1 batch regardless of gradient accum)
    end = time.time()

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
            max_length=128,
            padding="longest",
            truncation="only_first",
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
            st_idx = (answer_labels[bs]==question_token_id).nonzero(as_tuple=True)[0]
            ed_idx = (answer_labels[bs]==answer_token_id).nonzero(as_tuple=True)[0]
            answer_labels[bs,st_idx:ed_idx] = -100

        labels.to(device_id)
        answer_labels.to(device_id)

        with autocast():
            loss = model(
                vision_x=input_imgs,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=answer_labels,
                use_local_vision=use_local_vision,
            )[0]
        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()

        #### MASK GRADIENTS FOR EMBEDDINGS ####
        # Note (anas): Do not apply weight decay to embeddings as it will break this function.
        def mask_embedding(m):
            if isinstance(m, torch.nn.Embedding) and m.weight.requires_grad:
                zero_mask = torch.zeros_like(m.weight.grad)
                zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
                zero_mask[endofchunk_token_id] = torch.ones_like(
                    zero_mask[endofchunk_token_id]
                )
                m.weight.grad = m.weight.grad * zero_mask

        model.apply(mask_embedding)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if tb_log is not None:
                try:
                    cur_lr = float(optimizer.lr)
                except:
                    cur_lr = optimizer.param_groups[0]['lr']

                tb_log.add_scalar('meta_data/learning_rate',cur_lr,global_step)
                tb_log.add_scalar('train/loss', loss.item(), global_step)

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            logger.info(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}"
            )

    return global_step


def val_one_epoch(args,model,epoch,data_loader,tokenizer,optimizer,lr_scheduler,device_id,tb_log=None,logger=None):
    model.eval()

    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    question_token_id = tokenizer("Question", add_special_tokens=False)["input_ids"][-1]
    answer_token_id = tokenizer("?Answer", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    loss_metric = Metrics()

    with torch.no_grad():
        for idx, batch_dict in tqdm(
                enumerate(data_loader),
                total=data_loader.num_batches,
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
                answer_labels[bs, st_idx:ed_idx] = -100

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

    val_loss = loss_metric.average()
    return val_loss

def main():
    args = read_args()

    ############# CONFIGURATION #############
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    global_cfg.Dataset.Img_Features_File_Map = global_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    global_cfg.Dataset.Object_Features_File_Map = global_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    args.enable_imgdataset = False if global_cfg.Dataset.get('IMG_DIR',None) is None else True

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
    )

    if args.split == 'train':
        logger.info("**************************** Training ****************************")

        ############# DATASET #############
        tokenizer.padding_side = "right"
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

        if args.trainval_step > 0:
            val_dataset = BaseDataset(
                config=global_cfg.Dataset,
                split="val_seen",
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

        if args.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_model = DDP(model, device_ids=[device_id])
            # args.batch_size: BATCH_SIZE_PER_GPU
            logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
        else:
            total_gpus = 1
            ddp_model = model
            logger.info('Training with a single process')

        optimizer = torch.optim.AdamW(get_grouped_params(ddp_model, args), lr=args.learning_rate)

        total_training_steps = (
                                       len(dataset) // (args.batch_size * args.world_size)
                               ) * args.num_epochs

        logger.info(f"Total training steps: {total_training_steps}")
        tb_log = SummaryWriter(log_dir=str(Path(args.run_name) / 'tensorboard')) if args.rank == 0 else None

        if args.lr_scheduler == "linear":
            raise NotImplementedError
        elif args.lr_scheduler == "cosine":
            raise NotImplementedError
        else:
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        # TODO : check if a checkpoint exists for this run
        resume_from_epoch, global_step = check_checkpoint(args, ddp_model, optimizer, lr_scheduler, logger)

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

            if args.trainval_step > 0:
                if epoch % args.trainval_step == 0:
                    val_loss = val_one_epoch(
                        args=args,
                        model=ddp_model,
                        epoch=epoch,
                        data_loader=val_dataloader,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        device_id=device_id,
                        tb_log=tb_log,
                        logger=logger
                    )
                    logger.info("[Training with Validation Loss {:.2f}]".format(val_loss))
                else:
                    val_loss = 0
            else:
                val_loss = 0

            if args.rank == 0 and min_val_loss >= val_loss:
                min_val_loss = val_loss
                # save checkpoint
                save_checkpoint(args, epoch, ddp_model, optimizer, lr_scheduler, logger, global_step, min_val_loss)

        if args.rank == 0:
            # save final weights
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)
            torch.save(get_checkpoint(ddp_model), f"{args.run_name}/final_weights.pt")

    elif 'val' in args.split: # Single-GPU Val
        logger.info("**************************** Validation: {} ****************************".format(args.split))

        model = model.to(device_id)

        validation(
            args=args,
            global_cfg=global_cfg,
            model=model,
            tokenizer=tokenizer,
            device_id=device_id.index
        )
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()