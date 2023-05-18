import torch
import numpy as np
import re
import random
import math
from tqdm import tqdm
from PIL import Image
import torchvision
from dataset.preprocess_data import promptQAs
from .train.train_utils import get_autocast, get_cast_dtype, AverageMeter
import torch.nn.functional as F
from dataset.process_multi_data import batch_process_text, batch_process_image
from tools.common_utils import all_gather

class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        return self.total / self.num


def get_new_obs(batch_dict):
    batch_env = batch_dict['env']
    batch_size = batch_dict['batch_size']
    batch_obs = []
    for bs in range(batch_size):
        target_vp = batch_dict['paths'][bs][-1]
        gt_path = batch_dict['paths'][bs]
        ob = batch_env[bs].get_obs(target_vp,gt_path)[0]
        assert batch_dict['scan'][bs] == ob['scan']
        ob.update({
            'instr_id': batch_dict['instr_id'][bs],
            'instruction': batch_dict['instruction'][bs],
            'gt_path': batch_dict['paths'][bs],
            'path_id': batch_dict['path_id'][bs],
            'sample_idx': batch_dict['sample_idx'][bs]
        })
        batch_obs.append(ob)
    return batch_obs


def reset_env(batch_dict):
    batch_env = batch_dict['env']
    batch_size = batch_dict['batch_size']
    batch_obs = []
    for bs in range(batch_size):
        target_vp = batch_dict['paths'][bs][-1]
        gt_path = batch_dict['paths'][bs]
        ob = batch_env[bs].get_obs(target_vp,gt_path)[0]
        assert batch_dict['scan'][bs] == ob['scan']
        ob.update({
            'instr_id': batch_dict['instr_id'][bs],
            'instruction': batch_dict['instruction'][bs],
            'gt_path': batch_dict['paths'][bs],
            'path_id': batch_dict['path_id'][bs],
            'sample_idx': batch_dict['sample_idx'][bs]
        })
        batch_obs.append(ob)

    # # gt path: next viewpoint
    # gt_viewpoints = []
    # for bi in range(batch_size):
    #     gt_vp = dict()
    #     for vi, vp in enumerate(batch_obs[bi]['gt_path']):
    #         if vp == batch_obs[bi]['gt_path'][-1]:
    #             gt_vp[vp] = None
    #         else:
    #             gt_vp[vp] = batch_obs[bi]['gt_path'][vi + 1]
    #     gt_viewpoints.append(gt_vp)

    return batch_obs


def compute_label(obs,ended):
    """
    Args:
        obs:
        ended:

    Returns:
        answers: [viewpoint/'stop'/None]
    """
    answers = []
    for i, ob in enumerate(obs):
        if ended[i]:  # Just ignore this index
            gt_ans = None # ignore
        else:
            if ob['viewpoint'] == ob['teacher']:
                # the teacher is the current location, just stop
                gt_ans = 'stop'
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate == ob['teacher']:  # Next view point
                        gt_ans = str(ob['candidate_view_id'][candidate])
                        break
                    # else:  # Stop here
                    #     assert ob['teacher'] == ob['viewpoint']  # The teacher action should be "STAY HERE"
                    #     gt_ans = 'stop'
        answers.append(gt_ans)

    # answers = []
    # for bs in range(len(obs)):
    #     cur_viewpoint = cur_vps[bs]
    #     if cur_viewpoint == target_vps[bs]: # STOP
    #         gt_next_viewpoint = None
    #         answer = 'stop'
    #     else:
    #         if cur_viewpoint in gt_viewpoints[bs].keys():
    #             gt_next_viewpoint = gt_viewpoints[bs][cur_viewpoint]
    #         else:
    #             # fint the shortest path: current viewpoint -> target viewpoint
    #             shortest_path = shortest_paths[obs[bs]['scan']][cur_viewpoint][target_vps[bs]]
    #             gt_next_viewpoint = shortest_path[
    #                 shortest_path.index(cur_viewpoint)+1
    #             ]
    #         answer = str(obs[bs]['candidate_view_id'][gt_next_viewpoint])
    #     answers.append(answer)
    return answers


def process_text(
        obs, answers, tokenizer, t,
        agent_config, args, device_id, cast_dtype
):
    batch_input_text = []
    direction_ids = [str(_) for _ in range(12)]

    prompt = "{task_description}{instruction}{history}" \
             "{environment}{question}{answer}{endofchunk}{tokenizer_eos_token}"

    task_description = "Task:You are a mobile agent in an indoor building." \
                       "I will provide 12 images of the environment from different direction angles." \
                       "You need to complete the given instruction," \
                       "and you can use <walkto0>,<walkto1>,<walkto2>,<walkto3>,<walkto4>,<walkto5>," \
                       "<walkto6>,<walkto7>,<walkto8>,<walkto9>,<walkto10>,<walkto11> to explore the environment " \
                       "or use <stop> to stop."

    environment = "#Step {},the environment is ".format(t+1) \
                  + "".join(['{}<image>'.format(x,x) for x in range(12)])

    question = "#Question:which direction does the navigation instruction refer to"
    for bs in range(len(obs)):
        instr = "#Instruction:{}".format(obs[bs]['instruction'])

        if answers[bs] is None:
            answer = "#Answer:<Done>"
        elif answers[bs] == 'stop':
            answer = "#Answer:<stop>"
        elif answers[bs] in direction_ids:
            answer = "#Answer:<walkto{}>".format(answers[bs])
        else:
            raise NotImplementedError

        if args.multi_state:
            history_state = "".join(["{}<state>".format(_) for _ in range(t+1)])
            state = "#The visited environment state is {}.".format(history_state)
        else:
            state = "#The visited environment state is {}.".format('<state>')
        input_text = prompt.format(
            task_description=task_description,
            instruction=instr,
            history=state,
            environment=environment,
            question=question,
            answer=answer,
            endofchunk='<|endofchunk|>', # empty
            tokenizer_eos_token='</s>' # empty
        )

        input_text = (
            input_text.replace(" <|endofchunk|>", "<|endofchunk|>")
            .replace("<image> ", "<image>")
            .replace(" <image>", "<image>")
            .replace(" Question", ".Question")
            .replace(" Answer", "?Answer")
        )
        batch_input_text.append(input_text)

    batch_input_text = tokenizer(
        batch_input_text,
        max_length=args.max_length,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )

    input_ids = batch_input_text['input_ids'].to(device_id, dtype=cast_dtype, non_blocking=True)
    attention_mask = batch_input_text['attention_mask'].to(
        device_id, dtype=cast_dtype, non_blocking=True
    )

    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100  # <PAD> LLaMa
    labels[:, 0] = -100  # first token <s> or <PAD>
    labels[labels == args.media_token_id] = -100

    # question->answer:
    answer_labels = labels.clone()

    # Shift so that tokens < n predict n logits
    answer_locs = []
    for bs in range(labels.shape[0]):
        # LLaMa: tokenizer.decode([0,1,2,32000,32001,32002])
        #   >>> '<unk><s></s><|endofchunk|><image><PAD>'
        # st_idx = (answer_labels[bs] == args.question_token_id).nonzero(as_tuple=True)[0]

        if answers[bs] is None: # ignore all
            answer_labels[bs, :] = -100
            answer_locs.append(0)
        else:
            ed_idx = (answer_labels[bs] == args.answer_token_id).nonzero(as_tuple=True)[0]
            ed_idx += 2  # "#Answer:"
            answer_labels[bs, :ed_idx] = -100
            answer_locs.append(ed_idx-1) # shift left

    answer_labels = answer_labels.to(device_id, dtype=cast_dtype, non_blocking=True)

    return input_ids,attention_mask,answer_labels,answer_locs


def process_image(batch_imgs,device_id,cast_dtype,training,image_preprocess):
    batch_images = []
    for bs in range(len(batch_imgs)):
        imgs = np.hsplit(batch_imgs[bs], 12)
        images = [
            image_preprocess(
                Image.fromarray(
                    s[:, :, ::-1]  # BRG2RGB
                )
            ).unsqueeze(0)
            for s in imgs
        ]
        images = torch.cat(images, dim=0)  # [12,3,224,224]
        if training:
            # apply random horizontal flip and color jitter
            images = torchvision.transforms.RandomHorizontalFlip(p=0.5)(images)
            images = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(images)
        batch_images.append(images)
    batch_images = torch.stack(batch_images, 0)

    input_imgs = batch_images.to(device_id, dtype=cast_dtype, non_blocking=True)
    input_imgs = input_imgs.unsqueeze(2)
    return input_imgs


def parse_predictions_to_actions(
        batch_size,
        ended,
        logits,
        answer_locs,
        answers,
        tokenizer,
        policy_log_probs,
        obs,
        agent_config,
        t,
        random_path: bool = False,
):
    """
    Args:
        batch_size:
        ended: [bool,...] True:finished, False:navigation is not over
        logits: prediction logits from LLM
        answer_locs: answer label positions
        answers: []
        tokenizer:
        policy_log_probs:
        obs: observations
        agent_config: configurations
        t: in step $t$
        random_path: True: random select next viewpoint, False: use GT path.

    Returns:
        actions: [viewpoint_1,...,or None] None is STOP.
    """
    actions = []
    direction_labels = ['<walkto{}>'.format(_) for _ in range(12)]
    answer_set = ['{}'.format(_) for _ in range(12)]
    for bs in range(batch_size):
        if ended[bs]:  # if early stop, next action is None[STOP]
            actions.append(None)
            continue
        answer_logits = logits[bs, answer_locs[bs]]  # remove <endofchunk> </s>
        pred_tokens = torch.argmax(answer_logits, dim=-1)
        pred_txts = tokenizer.decode(pred_tokens, skip_special_tokens=False)

        # for RL-learning
        log_probs = F.log_softmax(logits[bs, answer_locs[bs]], 1)
        policy_log_probs.append(log_probs[:, pred_tokens[0]])

        # parse predictions
        if pred_txts == '<stop>':
            actions.append(None)
            ended[bs] = True
            continue
        elif pred_txts in direction_labels:
            pred_view_id = direction_labels.index(pred_txts)
            view_id_candidate = list(obs[bs]['view_id_candidate'].keys())
            # find the minimum distance from the pred view id.
            next_view_id = min(view_id_candidate, key=lambda x: abs(x - pred_view_id))
            actions.append(obs[bs]['view_id_candidate'][next_view_id])
        else:
            if random_path:
                # if pred failed, random select next view point.
                candidate = list(obs[bs]['candidate'].keys()) + [None]  # add [STOP]
                weights = [agent_config.max_action_len] * (len(candidate) - 1) + [t]
                random_action = random.choices(candidate, weights=weights)[0]
                if random_action is None:
                    actions.append(None)
                    ended[bs] = True
                else:
                    actions.append(random_action)
            else:
                # if pred failed, use GT path.
                if answers[bs] is None or answers[bs] == 'stop':
                    actions.append(None)
                    ended[bs] = True
                    continue
                elif answers[bs] in answer_set:
                    next_view_id = int(answers[bs])
                    actions.append(
                        obs[bs]['view_id_candidate'][next_view_id]
                    )
                else:
                    raise NotImplementedError

    return actions


def make_equiv_action(actions, obs, traj, batch_env):
    for i, ob in enumerate(obs):
        action = actions[i]
        if action is not None:
            traj[i]['pred_path'].append(action)
            prev_vp = traj[i]['pred_path'][-2]

            next_vp = batch_env[i].navigable_loc[prev_vp][action]
            next_view_id = next_vp['pointId']
            heading = (next_view_id % 12) * math.radians(30)
            # elevation = (next_view_id // 12 - 1) * math.radians(30)
            batch_env[i].msims[0].newEpisode(ob['scan'], action, heading)
        else:
            # just stop
            continue


# def train_one_epoch(
#     args,model,agent_config,epoch,r2r_dataset,r2r_dataloader,tokenizer,optimizer,lr_scheduler,device_id,tb_log=None,logger=None
# ):
#     model.train()
#     loss_metric = Metrics()
#     autocast = get_autocast(args.precision)
#     cast_dtype = get_cast_dtype(args.precision)
#
#     num_batches_per_epoch = r2r_dataloader.num_batches
#     total_training_steps = num_batches_per_epoch * args.num_epochs
#
#     pbar = tqdm(
#         enumerate(r2r_dataloader),
#         disable=args.rank!=0,
#         total=total_training_steps,
#         initial=(epoch * num_batches_per_epoch)
#     )
#     for num_steps, batch_dict in pbar:
#         global_step = num_steps + epoch * num_batches_per_epoch
#
#         # [1] reset env
#         obs = reset_env(batch_dict)
#
#         # [2] Initialization the tracking state
#         batch_size = len(obs)
#         ended = np.array([False] * batch_size)
#         policy_log_probs = []
#         traj = [{
#             'instr_id': ob['instr_id'],
#             'pred_path': [ob['viewpoint']], # start location
#         } for ob in obs]
#
#         ########## Navigation Process ##########
#         traj_loss = 0.
#         avg_step_loss = Metrics()
#         for t in range(agent_config.max_action_len):
#             input_imgs = process_image(
#                 [ob['panoramic_img'] for ob in obs],
#                 device_id=device_id,
#                 cast_dtype=cast_dtype,
#                 training=r2r_dataset.training,
#                 image_preprocess=r2r_dataset.image_preprocess
#             )
#
#             # label: next action/viewpoint
#             answers = compute_label(obs,ended)
#
#             # input text:
#             input_ids, attention_mask, answer_labels, answer_locs = \
#                 process_text(
#                     obs, answers, tokenizer, t,
#                     agent_config, args, device_id, cast_dtype
#                 )
#
#             with autocast():
#                 outputs = model(
#                     vision_x=input_imgs,
#                     lang_x=input_ids,
#                     attention_mask=attention_mask,
#                     labels=answer_labels,
#                     history_vis=t,
#                 )
#                 loss = outputs[0]
#                 logits = outputs[1]
#             traj_loss += loss
#             if not args.single_step_loss:
#                 ######### Loss #########
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()
#             avg_step_loss.accumulate(loss.data.item())
#
#             ######### Make Action #########
#             actions = parse_predictions_to_actions(
#                 batch_size, ended, logits, answer_locs, answers,
#                 tokenizer, policy_log_probs,
#                 obs, agent_config, t
#             )
#
#             make_equiv_action(
#                 actions=actions,
#                 obs=obs,
#                 traj=traj,
#                 batch_env=batch_dict['env'],
#             )
#
#             obs = get_new_obs(batch_dict)
#
#             ended[:] = np.logical_or(ended, np.array([x is None for x in actions]))
#
#             # Early exit if all ended
#             if ended.all():
#                 break
#
#         if args.single_step_loss:
#             ######### Loss #########
#             traj_loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             loss_metric.accumulate(traj_loss.data.item())
#         else:
#             loss_metric.accumulate(traj_loss.data.item())
#
#         if tb_log is not None and args.rank == 0:
#             try:
#                 cur_lr = float(optimizer.lr)
#             except:
#                 cur_lr = optimizer.param_groups[0]['lr']
#
#             tb_log.add_scalar('meta_data/learning_rate', cur_lr, global_step)
#             tb_log.add_scalar('train/traj_loss', traj_loss.data.item(), global_step)
#             tb_log.add_scalar('train/avg_step_loss', avg_step_loss.average, global_step)
#
#             pbar.update()
#             pbar.set_postfix(dict(
#                 traj_loss=traj_loss.data.item(),
#                 avg_step_loss=avg_step_loss.average,
#                 step=global_step,
#                 lr=cur_lr,
#             ))
#
#         # Log loss to console
#         if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
#             logger.info(
#                 f"\nStep {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. "
#                 f"\nAverage Loss: {loss_metric.average:.3f}"
#             )
#
#     return global_step

def train_one_epoch(
    args,model,agent_config,epoch,r2r_dataset,r2r_dataloader,tokenizer,optimizer,lr_scheduler,device_id,tb_log=None,logger=None
):
    model.train()
    loss_metric = Metrics()
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    num_batches_per_epoch = r2r_dataloader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        enumerate(r2r_dataloader),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    for num_steps, batch_dict in pbar:
        global_step = num_steps + epoch * num_batches_per_epoch
        batch_size = batch_dict['batch_size']

        # [2] IMAGE # size: [B, T_img*M=12*M, 1, 3, 224, 224]
        input_image, image_mask, input_angle_feats = batch_process_image(
            batch_image=batch_dict['input_image'],
            batch_size=batch_size,
            batch_angle_feats=batch_dict['input_angle_feats']
        )
        input_image = input_image.to(device_id, dtype=cast_dtype, non_blocking=True)
        input_angle_feats = input_angle_feats.to(device_id, dtype=cast_dtype, non_blocking=True)

        # [1] TEXT
        input_ids, attention_mask, labels, image_mask = \
            batch_process_text(
                batch_dict=batch_dict,
                tokenizer=tokenizer,
                max_length=args.max_length,
                args=args,
                image_mask=image_mask,
            )
        image_mask = image_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

        input_ids = input_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = attention_mask.to(device_id, dtype=cast_dtype, non_blocking=True)
        labels = labels.to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast():
            outputs = model(
                vision_x=(input_image,image_mask,input_angle_feats),
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            logits = outputs[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loss_metric.accumulate(loss.data.item())

        if tb_log is not None and args.rank == 0:
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']

            tb_log.add_scalar('meta_data/learning_rate', cur_lr, global_step)
            tb_log.add_scalar('train/traj_loss', loss.data.item(), global_step)
            tb_log.add_scalar('train/avg_step_loss', loss_metric.average, global_step)

            pbar.update()
            pbar.set_postfix(dict(
                loss=loss.data.item(),
                loss_metric=loss_metric.average,
                step=global_step,
                lr=cur_lr,
            ))

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            logger.info(
                f"\nStep {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. "
                f"\nAverage Loss: {loss_metric.average:.3f}"
            )

    return global_step


def collect_pred_actions(batch_dict, batch_preds, rank, pred_dict):
    for bs in range(len(batch_preds)):
        key = '{}_{}'.format(
            batch_dict['data_type'][bs],
            batch_dict['instr_id'][bs]
        )
        if pred_dict.get(key, None) is None:
            pred_dict[key] = dict()
        pred_dict[key]['input_text'] = batch_dict['input_text'][bs]
        pred_dict[key]['pred_text'] = batch_preds[bs]

    all_pred_dict = all_gather(pred_dict)
    if rank == 0:
        for per_results in all_pred_dict:
            for k, v in per_results.items():
                pred_dict[k] = v

@torch.no_grad()
def evaluate(
    args, model, r2r_dataset, r2r_dataloader, tokenizer, device_id, logger=None
):
    model.eval()
    loss_metric = Metrics()
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    num_batches_per_epoch = r2r_dataloader.num_batches
    total_training_steps = num_batches_per_epoch * 1 # args.num_epochs

    results = {
        'r2r': 0, 'r2r_sum': 0,
        'cvdn': 0, 'cvdn_sum': 0,
        'soon': 0, 'soon_sum': 0,
        'reverie': 0, 'reverie_sum': 0,
        'true': 0, 'all': 0,
    }

    pbar = tqdm(
        enumerate(r2r_dataloader),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(0 * num_batches_per_epoch)
    )

    pred_dict = {}

    for num_steps, batch_dict in pbar:
        batch_size = batch_dict['batch_size']

        # [2] IMAGE # size: [B, T_img*M=12*M, 1, 3, 224, 224]
        input_image, image_mask, input_angle_feats = batch_process_image(
            batch_image=batch_dict['input_image'],
            batch_size=batch_size,
            batch_angle_feats=batch_dict['input_angle_feats']
        )
        input_image = input_image.to(device_id, dtype=cast_dtype, non_blocking=True)
        input_angle_feats = input_angle_feats.to(device_id, dtype=cast_dtype, non_blocking=True)

        # [1] TEXT
        input_ids, attention_mask, labels, image_mask = \
            batch_process_text(
                batch_dict=batch_dict,
                tokenizer=tokenizer,
                max_length=args.max_length,
                args=args,
                image_mask=image_mask,
            )
        image_mask = image_mask.to(device_id, dtype=cast_dtype, non_blocking=True)

        input_ids = input_ids.to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = attention_mask.to(device_id, dtype=cast_dtype, non_blocking=True)
        labels = labels.to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast():
            outputs = model(
                vision_x=(input_image,image_mask,input_angle_feats),
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs[0]
            logits = outputs[1]

        loss_metric.accumulate(loss.data.item())

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_preds = torch.argmax(shift_logits, dim=-1)

        sample_results = {
            'r2r': 0, 'r2r_sum': 0,
            'cvdn': 0, 'cvdn_sum': 0,
            'soon': 0, 'soon_sum': 0,
            'reverie': 0, 'reverie_sum': 0,
            'true': 0, 'all': 0,
        }

        pred_acts = []
        for bs in range(batch_size):
            data_type = batch_dict['data_type'][bs]

            # gt_action_index = (shift_labels[bs]!=-100).nonzero().view(-1)
            gt_action_index = \
                ((shift_labels[bs] <= args.action_token_ids[-1])
                 & (shift_labels[bs] >= args.action_token_ids[0])
                 ).nonzero().view(-1)

            # views_locs = (input_ids[bs] >= 32001) & (input_ids[bs] <= 32012)
            # action_locs = (input_ids[bs] >= args.action_token_ids[0]) & (input_ids[bs] <= args.action_token_ids[-1])
            # view_action_locs = views_locs | action_locs
            # view_actions_pairs = input_ids[bs,view_action_locs]

            gt_actions = shift_labels[bs][gt_action_index].detach().cpu().numpy()
            pred_actions = shift_preds[bs][gt_action_index].detach().cpu().numpy()
            assert ((gt_actions <= args.action_token_ids[-1]) &
                    (gt_actions >= args.action_token_ids[0])).any()

            sample_results['true'] += (pred_actions == gt_actions).sum()
            sample_results['all'] += gt_actions.shape[0]
            sample_results[data_type] += (pred_actions == gt_actions).sum()
            sample_results['{}_sum'.format(data_type)] += gt_actions.shape[0]

            pred_acts.append(pred_actions)

        # collect pred actions
        collect_pred_actions(
            batch_dict=batch_dict,
            batch_preds=tokenizer.batch_decode(pred_acts),
            rank=args.rank,
            pred_dict=pred_dict
        )

        batch_results = all_gather(sample_results)
        if args.rank == 0:
            for per_results in batch_results:
                for k,v in per_results.items():
                    results[k] += v

            pbar.update()
            pbar.set_postfix(dict(
                true_cases=results['true'],
                all_cases=results['all'],
                loss=loss_metric.average,
                acc=(results['true']/results['all']),
            ))

    if args.rank == 0:
        logger.info("[Eval] {} split: Pred/All = ({})/({}) = {:.2f}%, Val Loss: {:.2f}".format(
            args.split, results['true'], results['all'], (100*results['true']/results['all']), loss_metric.average
        ))
        for k, v in results.items():
            if 'sum' in k:
                continue
            if k == 'true' or k == 'all':
                continue
            logger.info(" - [{}] dataset: Pred/All = ({})/({}) = {:.2f}%".format(
                k, results[k], results['{}_sum'.format(k)], (100*results[k]/(results['{}_sum'.format(k)]+1))
            ))

        from pathlib import Path
        import datetime
        import json
        val_pred_file = Path(args.run_name) / (
            '{}_pred_{}.json'.format(
                args.split,
                datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            ))
        with open(str(val_pred_file), 'w') as f:
            json.dump(pred_dict, f, indent=2)



###########################################################################################
    #     # [1] reset env
    #     obs = reset_env(batch_dict)
    #
    #     # [2] Initialization the tracking state
    #     batch_size = len(obs)
    #     ended = np.array([False] * batch_size)
    #     policy_log_probs = []
    #     traj = [{
    #         'instr_id': ob['instr_id'],
    #         'pred_path': [ob['viewpoint']], # start location
    #     } for ob in obs]
    #
    #     ########## Navigation Process ##########
    #     traj_loss = 0.
    #     avg_step_loss = Metrics()
    #     for t in range(agent_config.max_action_len):
    #         input_imgs = process_image(
    #             [ob['panoramic_img'] for ob in obs],
    #             device_id=device_id,
    #             cast_dtype=cast_dtype,
    #             training=r2r_dataset.training,
    #             image_preprocess=r2r_dataset.image_preprocess
    #         )
    #
    #         # label: next action/viewpoint
    #         answers = compute_label(obs,ended)
    #
    #         # input text:
    #         input_ids, attention_mask, answer_labels, answer_locs = \
    #             process_text(
    #                 obs, answers, tokenizer, t,
    #                 agent_config, args, device_id, cast_dtype
    #             )
    #
    #         with autocast():
    #             outputs = model(
    #                 vision_x=input_imgs,
    #                 lang_x=input_ids,
    #                 attention_mask=attention_mask,
    #                 labels=answer_labels,
    #                 history_vis=t,
    #             )
    #             loss = outputs[0]
    #             logits = outputs[1]
    #         traj_loss += loss
    #         if not args.single_step_loss:
    #             ######### Loss #########
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()
    #             lr_scheduler.step()
    #             optimizer.zero_grad()
    #         avg_step_loss.accumulate(loss.data.item())
    #
    #         ######### Make Action #########
    #         actions = parse_predictions_to_actions(
    #             batch_size, ended, logits, answer_locs, answers,
    #             tokenizer, policy_log_probs,
    #             obs, agent_config, t
    #         )
    #
    #         make_equiv_action(
    #             actions=actions,
    #             obs=obs,
    #             traj=traj,
    #             batch_env=batch_dict['env'],
    #         )
    #
    #         obs = get_new_obs(batch_dict)
    #
    #         ended[:] = np.logical_or(ended, np.array([x is None for x in actions]))
    #
    #         # Early exit if all ended
    #         if ended.all():
    #             break
    #
    #     if args.single_step_loss:
    #         ######### Loss #########
    #         traj_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         loss_metric.accumulate(traj_loss.data.item())
    #     else:
    #         loss_metric.accumulate(traj_loss.data.item())
    #
    #     if tb_log is not None and args.rank == 0:
    #         try:
    #             cur_lr = float(optimizer.lr)
    #         except:
    #             cur_lr = optimizer.param_groups[0]['lr']
    #
    #         tb_log.add_scalar('meta_data/learning_rate', cur_lr, global_step)
    #         tb_log.add_scalar('train/traj_loss', traj_loss.data.item(), global_step)
    #         tb_log.add_scalar('train/avg_step_loss', avg_step_loss.average, global_step)
    #
    #         pbar.update()
    #         pbar.set_postfix(dict(
    #             traj_loss=traj_loss.data.item(),
    #             avg_step_loss=avg_step_loss.average,
    #             step=global_step,
    #             lr=cur_lr,
    #         ))
    #
    #     # Log loss to console
    #     if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
    #         logger.info(
    #             f"\nStep {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. "
    #             f"\nAverage Loss: {loss_metric.average:.3f}"
    #         )
    #
    # return global_step