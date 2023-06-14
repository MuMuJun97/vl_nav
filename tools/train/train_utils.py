import time
from contextlib import suppress
import torch
from tqdm import tqdm
import glob
import os

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_grouped_params(model,args):
    params_with_wd, params_without_wd = [], []

    for n, p in model.named_parameters():
        if p.requires_grad:
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)

    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def check_checkpoint(args, model, optimizer, lr_scheduler, logger, is_duet=False):
    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        if is_duet:
            checkpoint_list = glob.glob(f"{args.run_name}/best_val_unseen*")
            args.resume_from_checkpoint = checkpoint_list[-1]
            logger.info(
                f"Found checkpoint {args.resume_from_checkpoint} for \n run_name = {args.run_name}."
            )
        else:
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
    if is_duet:
        resume_from_epoch = global_step = 0
        if args.resume_from_checkpoint is not None:
            if args.rank == 0:
                logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")['vln_bert']
            model_state_dict = model.state_dict()
            state_disk = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            update_model_state = {}
            for key, val in state_disk.items():
                if key in model_state_dict and model_state_dict[key].shape == val.shape:
                    update_model_state[key] = val
                else:
                    logger.info(
                        'Ignore weight %s: %s' % (key, str(val.shape))
                    )
            model.load_state_dict(update_model_state, strict=False)
            resume_from_epoch = checkpoint["epoch"] + 1
            logger.info("Load epoch: resume from epoch {}".format(resume_from_epoch))
            return resume_from_epoch, global_step

    else:
        # TODO : resume from checkpoint
        resume_from_epoch = global_step = 0
        if args.resume_from_checkpoint is not None:
            if args.rank == 0:
                logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")

            model_state_dict = model.state_dict()
            state_disk = {k.replace('module.',''):v for k,v in checkpoint["model_state_dict"].items()}

            update_model_state = {}
            for key, val in state_disk.items():
                if key in model_state_dict and model_state_dict[key].shape == val.shape:
                    update_model_state[key] = val
                else:
                    logger.info(
                        'Ignore weight %s: %s' % (key, str(val.shape))
                    )
            model.load_state_dict(update_model_state,strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) if optimizer is not None else None
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"]) if lr_scheduler is not None else None
            resume_from_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
        return resume_from_epoch, global_step


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

def get_checkpoint(model, del_grad=True):
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model_state_to_cpu(model.module.state_dict())
            if del_grad:
                for name, p in model.named_parameters():
                    sn = name.replace('module.', '')
                    if not p.requires_grad:
                        del state_dict[sn]
        else:
            state_dict = model.state_dict()
            if del_grad:
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        del state_dict[name]
    else:
        state_dict = None

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
                os.remove(f"{args.run_name}/checkpoint_{epoch - args.save_ckpt_step}.pt")
