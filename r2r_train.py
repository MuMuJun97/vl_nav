import os
import pickle
import random
import re
import numpy as np
from copy import deepcopy
from PIL import Image
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from dataset.base_dataset import BaseDataset, build_dataloader
from dataset.environment import R2RNavBatch
from transformers import get_constant_schedule_with_warmup
from open_flamingo import create_model_and_transforms
from tools.parser import read_args,random_seed
from tools.train.train_utils import get_autocast, get_cast_dtype, AverageMeter
from tools.train.distributed import world_info_from_env, init_distributed_device
from tools.train.train_utils import (
    get_grouped_params, check_checkpoint,
    get_checkpoint, save_checkpoint,
)
import datetime
from dataset.preprocess_data import promptQAs


def main():
    args = read_args()

    ############# CONFIGURATION #############
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    global_cfg.Dataset.Img_Features_File_Map = global_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    global_cfg.Dataset.Object_Features_File_Map = global_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    args.enable_imgdataset = False if global_cfg.Dataset.get('IMG_DIR',None) is None else True
    args.max_length = global_cfg.Dataset.tokenizer.max_length

    device_id = init_distributed_device(args)  # TODO multi-GPU training.

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

    logger.info("**************************** Eval ****************************")

    ############# DATASET #############
    r2rdataset = R2RNavBatch(config=global_cfg.Dataset)

    # TODO dataset, dataloader, sampler

    random_seed(args.seed, args.rank)
    print(f"Start running training on rank {args.rank}.")

    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus
    model = model.to(device_id)
    optimizer = torch.optim.AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    # TODO : check if a checkpoint exists for this run

    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[device_id])
        # args.batch_size: BATCH_SIZE_PER_GPU
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        total_gpus = 1
        ddp_model = model
        logger.info('Training with a single process')

    ############# MODEL-TRAIN #############
    ddp_model.train()

    episode_len = 15
    lengths = 1000
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    predictions = {}
    oracle_acc = 0
    success_acc = 0
    acc_sum = 0
    pbar = tqdm(range(lengths),desc="inference navigation |",total=lengths)
    for i in pbar:
        obs = r2rdataset.reset()
        batch_size = len(obs)
        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [ob['viewpoint']],
            'gt_path': ob['gt_path'],
        } for ob in obs]

        gt_viewpoints = []
        for bi in range(batch_size):
            gt_vp = dict()
            for vi,vp in enumerate(obs[bi]['gt_path']):
                if vp == obs[bi]['gt_path'][-1]:
                    gt_vp[vp] = None
                else:
                    gt_vp[vp] = obs[bi]['gt_path'][vi+1]
            gt_viewpoints.append(gt_vp)

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        episode_pred = {obs[bs]['instr_id']:{} for bs in range(batch_size)}
        #################### Iterate ####################
        for t in range(episode_len):

            ########## [1] TEXT ##########
            batch_text = []
            for bs in range(batch_size):
                cur_text = promptQAs["task_description"]["short"] \
                           + "Environment:" \
                           + "".join(["<image>-direction {};".format(i) for i in range(12)]) \
                           + ".Question:" \
                           + "which direction does the navigation instruction \"{Instruction}\" refer to?".format(
                                Instruction=obs[bs]['instruction']) \
                           + "Answer:"
                           # input("Question: ")
                batch_text.append(cur_text)

                ########## [2] Label ##########
                cur_viewpoint = obs[bs]['viewpoint']
                cur_scan = obs[bs]['scan']

                # gt_next_viewpoint = gt_viewpoints[bs][cur_viewpoint] # gt_path
                # gt_next_viewpoint = obs[bs]['gt_path'][t] # step->gt_path
                if cur_viewpoint in obs[bs]['gt_path']:
                    gt_next_viewpoint = gt_viewpoints[bs][cur_viewpoint]  # gt_path
                else:
                    shortest_distances = r2rdataset.shortest_distances[cur_scan]
                    nearest_position = r2rdataset.get_nearest(
                        shortest_distances,
                        goal_id=obs[bs]['gt_path'][-1],
                        path=list(obs[bs]['candidate'].keys())
                    )
                    gt_next_viewpoint = nearest_position
                    if nearest_position == obs[bs]['gt_path'][-1]:
                        gt_next_viewpoint = None

                if gt_next_viewpoint is None:
                    gt_next_view_id = 'stop'
                else:
                    gt_next_view_id = obs[bs]['candidate_view_id'][gt_next_viewpoint] # direction id.
                answer = gt_next_view_id
                # print(batch_text[bs]+answer) if t == 0 else None

            input_text = tokenizer(
                batch_text,
                max_length=args.max_length,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            input_ids = input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = input_text['attention_mask'].to(
                device_id, dtype=cast_dtype, non_blocking=True
            )

            ########## [3] IMAGE ##########
            batch_image = []
            for bs in range(batch_size):
                imgs = np.hsplit(obs[bs]['panoramic_img'], 12)
                images = [
                    image_processor(
                        Image.fromarray(
                            s[:, :, ::-1]  # BRG2RGB
                        )
                    ).unsqueeze(0)
                    for s in imgs
                ]
                images = torch.cat(images, dim=0) # [B,3,224,224]
                batch_image.append(images)
            batch_image = torch.stack(batch_image, 0)
            batch_image = batch_image.to(device_id, dtype=cast_dtype, non_blocking=True)
            batch_image = batch_image.unsqueeze(2) # [B,12,1,3,224,224]

            ########### [4] Forward ###########
            # with autocast():
            #     output = model(
            #         vision_x=batch_image,
            #         lang_x=input_ids,
            #         attention_mask=attention_mask,
            #         labels=None,
            #     )
            #     loss = None
            #     logits = output[0]

            ########### [4] Generate ###########
            with torch.inference_mode():
                outputs = model(
                    vision_x=batch_image,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    mode='generate',
                )
            outputs = outputs[:, len(input_ids[0]):]
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            ########## [5] MakeAction ##########
            pred_viewpoint = []
            for bs in range(batch_size):
                if ended[bs] or (t == episode_len - 1):
                    pred_viewpoint.append(None)
                    ended[bs] = True
                    continue
                else:
                    if 'stop' in outputs[bs]:
                        pred_viewpoint.append(None)
                        ended[bs] = True

                        # single-step inference: success
                        if answer == 'stop':
                            episode_pred[obs[bs]['instr_id']][t] = 1.0
                        continue
                    pred_view_id = [int(s) for s in re.findall(r'\b\d+\b', outputs[bs])]

                    if len(pred_view_id) == 0 or pred_view_id[0] > 11 or pred_view_id[0] < 0:
                        # if pred failed, random select next view point.
                        candidate = list(obs[bs]['candidate'].keys()) + [None]
                        random_action = random.choice(candidate)
                        if random_action is None:
                            ended[bs] = True
                        pred_viewpoint.append(random_action)
                        continue

                    pred_view_id = pred_view_id[0]
                    view_id_candidate = list(obs[bs]['view_id_candidate'].keys())

                    # find the minimum distance from the pred view id.
                    next_view_id = min(view_id_candidate, key=lambda x: abs(x - pred_view_id))
                    pred_viewpoint.append(obs[bs]['view_id_candidate'][next_view_id])

                    # single-step inference: success
                    if answer == next_view_id:
                        episode_pred[obs[bs]['instr_id']][t] = 1.0


            r2rdataset.make_equiv_action(
                pred_viewpoint=pred_viewpoint,obs=obs,traj=traj
            )

            obs = r2rdataset.get_obs()

            ended[:] = np.logical_or(ended, np.array([x is None for x in pred_viewpoint]))

            # Early exit if all ended
            if ended.all():
                break

        #################### Eval Item ####################
        for bs in range(batch_size):
            scores = r2rdataset.eval_item(
                scan=obs[bs]['scan'],
                pred_path=traj[bs]['path'],
                gt_path=traj[bs]['gt_path']
            )
            predictions[traj[bs]['instr_id']] = dict()
            predictions[traj[bs]['instr_id']].update({
                'pred_path': traj[bs]['path'],
                'gt_path': traj[bs]['gt_path'],
                'score': scores,
                'scan': obs[bs]['scan'],
                'instruction': obs[bs]['instruction'],
                'path_id': obs[bs]['path_id'],
                'single_step_inference': episode_pred[traj[bs]['instr_id']],
            })
            oracle_acc += scores['oracle_success']
            success_acc += scores['success']
            acc_sum += 1
        pbar.set_postfix({
            'oracle_success': "acc {:.2f} % ({}/{})".format(100*(oracle_acc/acc_sum),oracle_acc,acc_sum),
            'success_rate': "acc {:.2f} % ({}/{})".format(100 * (success_acc / acc_sum), success_acc, acc_sum),
        })
        pbar.update()
    print('[INFO] oracle_success acc: {:.2f} %'.format(100*(oracle_acc/acc_sum)))
    print('[INFO] success_rate acc: {:.2f} %'.format(100 * (success_acc / acc_sum)))


if __name__ == "__main__":
    main()