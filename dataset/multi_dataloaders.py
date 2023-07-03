import numpy as np
import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from dataset.dataset_src import SrcDataset, build_dataloader
from duet.map_nav_src_llm.utils.data import \
    ImageFeaturesDB, ObjectFeatureDB, load_obj2vps, SOONObjectFeatureDB


def create_multi_dataloaders(
        args,
        global_cfg,
        logger,
        tokenizer,
):
    dataloaders = {}

    ############# Dataset #############
    feat_db = ImageFeaturesDB(str(args.img_ft_file), args.image_feat_size)
    if args.use_object_feat:
        obj_feat_db = ObjectFeatureDB(str(args.obj_ft_file), args.obj_feat_size) # 768
        soon_obj_feat_db = SOONObjectFeatureDB(args.soon_ft_file, 2048)
    else:
        obj_feat_db = None
        soon_obj_feat_db = None
    obj2vps = load_obj2vps(str(args.obj_ft_file.parent.parent / 'annotations/BBoxes.json'))

    r2r_dataset = SrcDataset(
        config=global_cfg.Dataset,
        training=False if args.split != 'train' else True,
        logger=logger,
        args=args,
        feat_db=feat_db,
        tokenizer=tokenizer,
        test=False,
        obj_feat_db=(obj_feat_db, obj2vps, soon_obj_feat_db)
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
            obj_feat_db=(obj_feat_db, obj2vps, soon_obj_feat_db)
        )
        val_r2r_dataset, val_r2r_dataloader, val_r2r_sampler = build_dataloader(
            dataset=val_r2r_dataset,
            batch_size=args.batch_size,
            distributed=args.distributed,
            workers=args.workers,
            training=False
        )