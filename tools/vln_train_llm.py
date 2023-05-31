import os
import sys
import json
import numpy as np
import torch
import math
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from models.vln_model_llm import BertVLNModel
from torch.nn.utils.rnn import pad_sequence
from duet.map_nav_src_llm.utils.ops import pad_tensors, gen_seq_masks
from collections import defaultdict
# from duet.map_nav_src.networks.graph_utils import GraphMap
from duet.map_nav_src_llm.networks.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts
from duet.map_nav_src_llm.networks.ops import pad_tensors_wgrad
from duet.map_nav_src_llm.r2r.eval_utils import cal_dtw
from duet.map_nav_src_llm.utils.distributed import all_gather
from contextlib import nullcontext

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


class NavigationAgent(object):
    def __init__(self, args, shortest_distances, shortest_paths):
        self.args = args
        self.shortest_paths = shortest_paths
        self.shortest_distances = shortest_distances
        # buffer
        self.scanvp_cands = {}

    def get_instruction(self, item, data_type):
        if data_type == 'r2r':
            return 'Travel following the instruction, you can not ask for help. Instruction: ' \
                + item['instruction']
        elif data_type == 'soon':
            return 'Find the described target, you can not ask for help. Target: ' \
                + item['instruction']
        elif data_type == 'reverie':
            return 'Go to the location to complete the given task, you can not ask for help. Task: ' \
                + item['instruction']
        elif data_type == 'eqa':
            return 'Explore the scene and answer the question, you can not ask for help. Question: ' \
                          + item['instruction']
        elif data_type == 'cvdn':
            return 'Find the described target, you can ask for help. Target: ' \
                + item['instruction']

    def update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    def language_variable(self, obs, data_type):
        raw_instruction = []
        for ob, dt in zip(obs, data_type):
            raw_instruction.append(
                self.get_instruction(ob, dt)
            )
        return raw_instruction

    def panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                                 in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)  # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'cand_vpids': batch_cand_vpids,
        }

    def get_pos_fts(self, cnt_vp, cand_vps, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in cand_vps:
            rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cnt_vp, vp,
                base_heading=cur_heading, base_elevation=cur_elevation,
            )
            rel_angles.append([rel_heading, rel_elevation])
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return rel_ang_fts


    def nav_vp_variable(self, obs, start_pos, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i in range(len(obs)):
            cur_cand_pos_fts = self.get_pos_fts(
                obs[i]['position'], [cc['position']for cc in obs[i]['candidate']],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = self.get_pos_fts(
                obs[i]['position'], [start_pos[i]],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 8), dtype=np.float32)
            vp_pos_fts[:, :4] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 4:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }


    def teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.shortest_distances[scan],
                                        sum(traj[i]['path'], []) + self.shortest_paths[scan][ob['viewpoint']][vpid][1:],
                                        ob['gt_path'],
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':

                                    dist = self.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    # def make_equiv_action(self, a_t, gmaps, obs, traj=None, env=None):
    def make_equiv_action(self, a_t, obs, traj=None, env=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                # traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                traj[i]['path'].append([action])
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                env[i].sims[0].newEpisode([ob['scan']], [action], [heading], [elevation])


def vln_train_one_epoch(
        args,
        vln_model: BertVLNModel,
        vln_optimizer,
        language_model,
        language_optimizer,
        lr_scheduler,
        r2r_dataloader,
        epoch,
        nav_agent: NavigationAgent,
        logger,
):
    if args.enable_language_model:
        vln_model.train()
        language_model.train()
    else:
        vln_model.vln_bert.train()
        vln_model.critic.train()
        vln_bert_optimizer, critic_optimizer = vln_optimizer

    num_batches_per_epoch = r2r_dataloader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    pbar = tqdm(
        # enumerate(r2r_dataloader),
        range(len(r2r_dataloader)),
        disable=args.rank!=0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    dataloader_it = iter(r2r_dataloader)

    loss_metric = Metrics()
    entropy_metric = Metrics()
    for num_steps in pbar:
        global_step = num_steps + epoch * num_batches_per_epoch

        if args.enable_language_model:
            vln_optimizer.zero_grad()
            language_optimizer.zero_grad()
        else:
            vln_bert_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # one iteration:
        iter_loss = 0

        #################### imitation learning ####################
        feedback, train_ml, train_rl = 'teacher', 0.2, False
        # reset env
        try:
            batch_dict = next(dataloader_it)
        except StopIteration:
            dataloader_it = iter(r2r_dataloader)
            batch_dict = next(dataloader_it)
            logger.info("\n[Dataloader] new iters")

        ml_loss, _ = rollout(
            args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
            train_ml=train_ml, train_rl=train_rl, nav_agent=nav_agent, vln_model=vln_model,
            entropy_metric=entropy_metric,
        )
        if train_ml is not None:
            iter_loss += ml_loss
            loss_metric.accumulate(ml_loss.item())

        #################### dagger training ####################
        feedback, train_ml, train_rl = 'sample', 1, False

        # reset env
        try:
            batch_dict = next(dataloader_it)
        except StopIteration:
            dataloader_it = iter(r2r_dataloader)
            batch_dict = next(dataloader_it)
            logger.info("[Dataloader] new iters")

        sample_ml_loss, _ = rollout(
            args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
            train_ml=train_ml, train_rl=train_rl, nav_agent=nav_agent, vln_model=vln_model,
            entropy_metric=entropy_metric,
        )
        if train_ml is not None:
            iter_loss += sample_ml_loss
            loss_metric.accumulate(sample_ml_loss.item())

        # iter_loss.backward()

        if args.enable_language_model:
            raise NotImplementedError
        else:
            torch.nn.utils.clip_grad_norm_(vln_model.vln_bert.parameters(), 40.)
            vln_bert_optimizer.step()
            critic_optimizer.step()

        if args.rank == 0:
            # pbar.update()
            pbar.set_postfix(dict(
                loss=loss_metric.average,
                entropy=entropy_metric.average,
            ))


def rollout(
        args,
        r2r_dataloader,
        batch_dict,
        feedback,
        train_ml,
        train_rl,
        nav_agent: NavigationAgent,
        vln_model: BertVLNModel,
        entropy_metric,
        do_backward=False,
        is_val=False, # in validation mode
):
    obs = batch_dict['observations']
    envs = batch_dict['env']

    nav_agent.update_scanvp_cands(obs)
    batch_size = len(obs)

    # Record the navigation path
    traj = [{
        'instr_id': ob['instr_id'],
        'path': [[ob['viewpoint']]],
        'details': {},
    } for ob in obs]

    start_pos = [ob['position'] for ob in obs]

    # Language input: txt_ids, txt_masks
    instruction = nav_agent.language_variable(obs, data_type=batch_dict['data_type'])
    history = []
    hist_vis = []
    for idx in range(len(instruction)):
        history.append("")
        hist_vis.append([])

    # Initialization the tracking state
    ended = np.array([False] * batch_size)
    just_ended = np.array([False] * batch_size)

    # Init the logs
    entropys = []
    ml_loss = 0.
    cnt_loss = 0.
    flag = False

    for t in range(args.max_action_len):

        if isinstance(vln_model.vln_bert, torch.nn.parallel.DistributedDataParallel):
            # multi-gpu
            if ended.all() or t == args.max_action_len - 1:
                flag = True
                context = nullcontext
            else:
                context = vln_model.vln_bert.no_sync
        else:
            # single-gpu
            if ended.all() or t == args.max_action_len - 1:
                break
                context = nullcontext
            else:
                context = nullcontext

        with context():

            pano_inputs = nav_agent.panorama_feature_variable(obs)

            pano_embeds, pano_masks = vln_model.vln_bert('panorama', pano_inputs)

            # navigation policy
            nav_inputs = \
                nav_agent.nav_vp_variable(
                    obs, start_pos, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            nav_inputs['instruction'] = instruction
            nav_inputs['history'] = history
            nav_inputs['hist_vis'] = hist_vis

            nav_outs = vln_model.vln_bert('navigation', nav_inputs)

            nav_logits = nav_outs['local_logits']
            nav_vpids = nav_inputs['vp_cand_vpids']
            nav_probs = torch.softmax(nav_logits, 1)

            # Imitation Learning
            if train_ml is not None or feedback == 'gt':
                # Supervised training
                if args.dataset == 'r2r':
                    nav_targets = nav_agent.teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=None,
                        imitation_learning=(feedback == 'teacher'), t=t, traj=traj
                    )
                cnt_loss += vln_model.criterion(nav_logits, nav_targets) * train_ml / batch_size
                ml_loss += cnt_loss.detach()

                if not is_val:
                    cnt_loss.backward()
                cnt_loss = 0.

            # Determinate the next navigation viewpoint
            if feedback == 'teacher':
                a_t = nav_targets  # teacher forcing
            elif feedback == 'sample':
                try:
                    c = torch.distributions.Categorical(nav_probs.float())
                except Exception as e:
                    import ipdb
                    ipdb.set_trace()
                    print(e) # nan, -inf values, may caused from different data type.
                    raise NotImplementedError

                entropy_metric.accumulate(c.entropy().sum().item()) # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
            elif feedback == 'gt':
                a_t = nav_targets # force gt path
            else:
                print(feedback)
                sys.exit('Invalid feedback option')

            for idx in range(len(a_t)):
                if a_t[idx] == -100:
                    continue
                history[idx] += '<hist>'
                hist_vis[idx].append(nav_outs['vp_embeds'][idx][a_t[idx]])

            # Determine stop actions
            if feedback == 'teacher' or feedback == 'sample':  # in training
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or (t == args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])

            # Make action and get the new state
            nav_agent.make_equiv_action(cpu_a_t, obs, traj, env=envs)

            # get new observation and update graph
            obs = []
            for b_i in range(batch_size):
                obs.append(
                    r2r_dataloader.dataset.get_obs(
                        items=[batch_dict['item'][b_i]],
                        env=envs[b_i]
                    )[0]
                )
            nav_agent.update_scanvp_cands(obs)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # # Early exit if all ended
            # if ended.all():
            #     break
            if flag:
                break

    # if train_ml is not None:
    #     ml_loss = ml_loss # * train_ml / batch_size

    return ml_loss, traj


def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs


def get_results(pred_results, detailed_output=False):
    pred_output = []
    for k, v in pred_results.items():
        pred_output.append({'instr_id': k, 'trajectory': v['path']})
    return pred_output


@torch.no_grad()
def vln_val_one_epoch(
        args,
        vln_model: BertVLNModel,
        vln_optimizer,
        language_model,
        language_optimizer,
        r2r_dataloader,
        epoch,
        nav_agent: NavigationAgent,
        logger,
        best_val,
        lr_scheduler=None,
):
    feedback = 'argmax'
    use_dropout = False
    results = {}
    entropy_metric = Metrics()

    if args.enable_language_model:
        vln_model.eval()
        language_model.eval()
    else:
        vln_model.vln_bert.eval()
        vln_model.critic.eval()

    pbar = tqdm(
        # enumerate(r2r_dataloader),
        range(len(r2r_dataloader)),
        disable=args.rank != 0,
        total=len(r2r_dataloader),
        initial=0,
        desc="validation: ",
    )
    dataloader_it = iter(r2r_dataloader)

    # We rely on env showing the entire batch before repeating anything
    looped = False

    for num_steps in pbar:
        # reset env
        batch_dict = next(dataloader_it)

        ml_loss, traj = rollout(
            args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
            train_ml=None, train_rl=False, nav_agent=nav_agent, vln_model=vln_model,
            entropy_metric=entropy_metric, is_val=True
        )

        for s_traj in traj:
            if s_traj['instr_id'] in results:
                looped = True
            else:
                ml_loss = 0
                results[s_traj['instr_id']] = s_traj

        if looped:
            break

    # [MULTI-GPU] gather all prediction results from ALL GPU
    preds = get_results(results)
    all_preds = all_gather(preds)
    all_preds = merge_dist_results(all_preds)

    loss_str = "\n[Eval] {} epoch {}\n".format(args.val_split, epoch)
    if args.rank == 0:
        multi_prefixs = set([pdata['instr_id'].split('_')[0] for pdata in all_preds])
        useful_score_summary = None
        for prefix in multi_prefixs:
            one_preds = []
            for pdata in all_preds:
                if pdata['instr_id'].split('_')[0] == prefix:
                    one_preds.append(pdata)

            score_summary, _ = r2r_dataloader.dataset.eval_metrics(one_preds, logger)

            if prefix == 'r2r':
                useful_score_summary = score_summary
            loss_str += "\n [Eval] dataset=[{}] \n".format(prefix)
            for metric, val in score_summary.items():
                if metric == 'sr':
                    loss_str += '\n[Eval] ||| %s: %.2f' % (metric, val)
                else:
                    loss_str += ', %s: %.2f' % (metric, val)

        logger.info(loss_str)
        if useful_score_summary is not None:
            score_summary = useful_score_summary

        # select model by Success Rate
        if score_summary['sr'] >= best_val[args.val_split]['sr']:
            best_val[args.val_split]['spl'] = score_summary['spl']
            best_val[args.val_split]['sr'] = score_summary['sr']
            best_val[args.val_split]['state'] = 'Epoch %d %s' % (epoch, loss_str)

            save_ckpt_file = Path(args.run_name) / "best_{}".format(args.val_split)
            vln_model.save(epoch, str(save_ckpt_file),
                           vln_bert_optimizer=vln_optimizer[0],
                           critic_optimizer=vln_optimizer[1]
                           )



