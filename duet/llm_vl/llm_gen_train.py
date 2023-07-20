import os
import torch
import datetime
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(Path(__file__).resolve().parent.parent.__str__())
sys.path.append(Path(__file__).resolve().parent.parent.parent.__str__())
from tools.parser import read_args, random_seed
from tools.train.distributed import init_distributed_device
from open_flamingo import create_model_and_transforms
from dataset.dataset_src import SrcDataset, build_dataloader
from duet.llm_vl.utils.data import \
    ImageFeaturesDB, ObjectFeatureDB, load_obj2vps, SOONObjectFeatureDB
from tools.finetune_utils import (get_tokenizer_token_ids, )
from tools.train.train_utils import (
    get_grouped_params, check_checkpoint,
    get_checkpoint, save_checkpoint,
)
from models.merge_model_llm import BertVLNModel
from tools.llm_gen_pipeline import vln_train_one_epoch, NavigationAgent, vln_val_one_epoch
from transformers import get_constant_schedule_with_warmup


def init_config(args):
    # single-gpu or multi-gpu
    device_id = init_distributed_device(args)

    ############# CONFIGURATION #############
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    global_cfg.Dataset.Img_Features_File_Map = global_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    global_cfg.Dataset.Object_Features_File_Map = global_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    args.enable_imgdataset = False if global_cfg.Dataset.get('IMG_DIR',None) is None else True
    args.max_length = global_cfg.Dataset.tokenizer.max_length
    args.max_action_len = global_cfg.Agent.max_action_len

    # offline image features, from vln-duet
    # downstream duet datasets
    _source_dir = Path(__file__).resolve().parent.parent.parent

    args.img_ft_file = _source_dir / "build/duet/R2R/features/view_timm_imagenet_vitb16"
    args.obj_ft_file = _source_dir / "build/duet/SOON/features/obj2d_ade20k_timm_vitb16"

    args.source_dir = _source_dir.__str__()
    args.vln_bert_pretrained_config = str(_source_dir / "tools/cfgs/vln_duet/vln_bert_pretrained_config.json")

    log_file = Path(args.run_name) / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    from tools import common_utils
    logger = common_utils.create_logger(log_file, rank=args.rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_config_to_file(global_cfg, logger=logger)

    return logger, global_cfg


def dist_models(args, vln_model: BertVLNModel, language_model=None, logger=None):
    logger.info("*************** init vln_model & language_model *************** ")
    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus
    resume_from_epoch = 0

    if args.enable_language_model:
        # @note: vln model + large language model
        vln_model = vln_model.to(device_id)
        language_model = language_model.to(device_id)
        vln_optimizer = torch.optim.AdamW(get_grouped_params(vln_model, args), lr=args.learning_rate)
        language_optimizer = torch.optim.AdamW(get_grouped_params(language_model, args), lr=args.learning_rate)
        lr_scheduler = get_constant_schedule_with_warmup(
            language_optimizer, num_warmup_steps=args.warmup_steps
        )

        # TODO : check if a checkpoint exists for this run
        # resume_from_epoch, global_step = check_checkpoint(args, model, optimizer, lr_scheduler, logger)

        if args.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            vln_model = DDP(vln_model, device_ids=[device_id])
            language_model = DDP(language_model, device_ids=[device_id])
            # args.batch_size: BATCH_SIZE_PER_GPU
            logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
        else:
            total_gpus = 1
            logger.info('Training with a single process')
    else:
        vln_model.vln_bert.to(device_id)
        vln_model.critic.to(device_id)
        vln_bert_optimizer = torch.optim.AdamW(vln_model.vln_bert.parameters(), lr=args.lr)
        critic_optimizer = torch.optim.AdamW(vln_model.critic.parameters(), lr=args.lr)

        language_optimizer = None
        lr_scheduler = None

        # TODO : check if a checkpoint exists for this run
        resume_from_epoch, global_step = check_checkpoint(
            args, vln_model.vln_bert, vln_bert_optimizer,
            lr_scheduler, logger, is_duet=True
        )
        vln_optimizer = (vln_bert_optimizer, critic_optimizer)

        if args.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            vln_model.vln_bert = DDP(vln_model.vln_bert, device_ids=[device_id], find_unused_parameters=True)
            vln_model.critic = DDP(vln_model.critic, device_ids=[device_id], find_unused_parameters=True)

            # args.batch_size: BATCH_SIZE_PER_GPU
            logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
        else:
            total_gpus = 1
            logger.info('Training with a single process')

    return vln_model, vln_optimizer, language_model, language_optimizer, resume_from_epoch, lr_scheduler


def main():
    """
    @file: duet/llm_vl/llm_training.py
    @function: VLN-DUET Baseline, use LLaMa-7B or other LLMs as language model.
    @param:
        args.enable_language_model = False: disable DUET Pipeline
                                   = True: enable DUET Pipeline
    """
    args = read_args()
    assert args.r2r_tok
    args.enable_language_model = False
    # enable validation when training
    args.val = True
    if args.val:
        args.val_split = 'val_unseen' # validation on val_unseen split

    logger, global_cfg = init_config(args)
    random_seed(seed=args.seed)

    ############# Language Model #############
    if args.enable_language_model:
        raise NotImplementedError
    else:
        args.seed = 0
        args.dataset = 'soon'

        args.enc_full_graph = True
        args.graph_sprels = True
        args.fusion = 'dynamic'

        args.image_feat_size = 768
        args.angle_feat_size = 4
        args.obj_feat_size = 768

        args.multi_startpoints = False
        args.multi_endpoints = True
        args.max_objects = 70
        args.max_action_len = 20

        args.num_l_layers = 9
        args.num_pano_layers = 2
        args.num_x_layers = 4

        args.ignoreid = -100
        args.dropout = 0.5
        args.expert_policy = 'spl'

        # we re-construct DUET Pipeline with LLMs
        vln_model = BertVLNModel(
            args, logger=logger, gen=True
        )

        # experiments
        if args.tokenizer_path == 'facebook/opt-iml-1.3b':
            param_sums = sum(p.numel() for p in vln_model.vln_bert.vln_bert.parameters() if p.requires_grad)
            print("OPT model initialized with {:.2f} M trainable parameters".format(param_sums/1000**2))
            # freeze opt weights
            # vln_model.vln_bert.vln_bert.lang_model.requires_grad_(False)
            # vln_model.vln_bert.vln_bert.lang_model.\
            #     lang_model.model.decoder.embed_tokens.requires_grad_(True)
            # vln_model.vln_bert.vln_bert.lang_model.\
            #     mapper.requires_grad_(True)
            param_sums = sum(p.numel() for p in vln_model.vln_bert.vln_bert.parameters() if p.requires_grad)
            print("after unfreeze: OPT model initialized with {:.2f} M trainable parameters".format(param_sums/1000**2))

        language_model = None
        tokenizer = None

    random_seed(args.seed + args.rank)

    ############# Dataset #############
    feat_db = ImageFeaturesDB(str(args.img_ft_file), args.image_feat_size)
    obj_db = ObjectFeatureDB(str(args.obj_ft_file), args.obj_feat_size)

    r2r_dataset = SrcDataset(
        config=global_cfg.Dataset,
        training=False if args.split != 'train' else True,
        logger=logger,
        args=args,
        feat_db=feat_db,
        tokenizer=tokenizer,
        test=False,
        obj_feat_db=obj_db
    )
    r2r_dataset, r2r_dataloader, r2r_sampler = build_dataloader(
        dataset=r2r_dataset,
        batch_size=args.batch_size,
        distributed=args.distributed,
        workers=args.workers,
        training=True
    )

    ####### val #######
    if args.val:
        val_r2r_dataset = SrcDataset(
            config=global_cfg.Dataset,
            training=False,
            logger=logger,
            args=args,
            feat_db=feat_db,
            tokenizer=tokenizer,
            test=False,
            split=args.val_split,
            obj_feat_db=obj_db
        )
        val_r2r_dataset, val_r2r_dataloader, val_r2r_sampler = build_dataloader(
            dataset=val_r2r_dataset,
            batch_size=args.batch_size,
            distributed=args.distributed,
            workers=args.workers,
            training=False
        )

    ############# Init DDP: VLN+Language Model #############
    print(f"Start running training on rank {args.rank}.")
    vln_model, vln_optimizer, \
        language_model, language_optimizer, \
        resume_from_epoch, lr_scheduler = \
        dist_models(
            args=args,
            vln_model=vln_model,
            language_model=language_model,
            logger=logger
        )

    ############# Agent: Step & Action #############
    nav_agent = NavigationAgent(
        args=args,
        shortest_distances=r2r_dataset.shortest_distances,
        shortest_paths=r2r_dataset.shortest_paths,
    )
    if args.val:
        val_nav_agent = NavigationAgent(
            args=args,
            shortest_distances=val_r2r_dataset.shortest_distances,
            shortest_paths=val_r2r_dataset.shortest_paths,
        )

    total_training_steps = (
                            len(r2r_dataset) // (args.batch_size * args.world_size)
                        ) * args.num_epochs

    logger.info(f"Total training steps: {total_training_steps}")
    tb_log = SummaryWriter(log_dir=str(Path(args.run_name) / 'tensorboard')) if args.rank == 0 else None

    logger.info("**************************** Train ****************************")
    best_val = {args.val_split: {"spl": 0., "sr": 0., "state": ""}}
    for epoch in range(resume_from_epoch, args.num_epochs):
        vln_train_one_epoch(
            args=args,
            vln_model=vln_model,
            vln_optimizer=vln_optimizer,
            language_model=language_model,
            language_optimizer=language_optimizer,
            lr_scheduler=None,
            r2r_dataloader=r2r_dataloader,
            epoch=epoch,
            nav_agent=nav_agent,
            logger=logger
        )
        if args.val:
            ########### validation ###########
            vln_val_one_epoch(
                args=args,
                vln_model=vln_model,
                vln_optimizer=vln_optimizer,
                language_model=language_model,
                language_optimizer=language_optimizer,
                lr_scheduler=None,
                r2r_dataloader=val_r2r_dataloader,
                epoch=epoch,
                nav_agent=val_nav_agent,
                logger=logger,
                best_val=best_val
            )
        if args.rank == 0:
            save_ckpt_file = Path(args.run_name) / "best_{}".format(args.val_split)
            vln_model.save(epoch, str(save_ckpt_file),
                           vln_bert_optimizer=vln_optimizer[0],
                           critic_optimizer=vln_optimizer[1]
                           )
            logger.info("\n[Best Result till Now]:\n{} | {}".format(args.val_split, best_val[args.val_split]['state']))


if __name__ == "__main__":
    main()