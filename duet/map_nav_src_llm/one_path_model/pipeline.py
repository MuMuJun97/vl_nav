import copy
import os
import sys
import json
import numpy as np
import torch
import math
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from duet.map_nav_src_llm.networks.one_path_model import BertVLNModel, prompt
from torch.nn.utils.rnn import pad_sequence
from duet.map_nav_src_llm.utils.ops import pad_tensors, gen_seq_masks
from collections import defaultdict
from duet.map_nav_src_llm.networks.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts
from duet.map_nav_src_llm.networks.ops import pad_tensors_wgrad
from duet.map_nav_src_llm.utils.distributed import all_gather
from contextlib import nullcontext
from networks.graph_utils import GraphMap

class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        if self.num == 0:
            return 0
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
            return 'Follow a given instruction to navigate the shortest path in an indoor environment. Instruction: ' \
                + item['instruction']
        elif data_type == 'soon':
            return 'Navigate to the location of an object described by a given instruction ' \
                   'in the shortest steps in an indoor environment. Instruction: ' \
                + item['instruction']
        elif data_type == 'reverie':
            return 'Navigate to a target location following a given instruction in an indoor environment. Instruction: ' \
                + item['instruction']
        elif data_type == 'eqa':
            return 'Answer a given question based on what you see in an indoor environment. Question: ' \
                + item['instruction']
        elif data_type == 'cvdn':
            return 'Navigate to the location of a target object in an indoor environment based on a dialogue: ' \
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

    def panorama_feature_variable_object(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_obj_img_fts, batch_loc_fts, batch_nav_types = [], [], [], []
        batch_view_lens, batch_obj_lens = [], []
        batch_cand_vpids, batch_objids = [], []

        have_objects = ['obj_img_fts' in ob.keys() and ob['obj_img_fts'] is not None for ob in obs]
        # if sum(have_objects) == len(obs):
        #     use_obj = 'both'
        # elif sum(have_objects) == 0:
        #     use_obj = 'none'
        # else:
        #     use_obj = 'single'

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

            batch_cand_vpids.append(cand_vpids)
            if 'eqa' in ob['instr_id']:
                batch_view_lens.append(5)
            else:
                batch_view_lens.append(len(view_img_fts))

            if have_objects[i]:
                # object
                obj_loc_fts = np.concatenate([ob['obj_ang_fts'], ob['obj_box_fts']], 1)
                nav_types.extend([2] * len(obj_loc_fts))
                batch_obj_img_fts.append(torch.from_numpy(ob['obj_img_fts']))
                batch_objids.append(ob['obj_ids'])
                batch_obj_lens.append(len(ob['obj_img_fts']))
                batch_loc_fts.append(torch.from_numpy(np.concatenate([view_loc_fts, obj_loc_fts], 0)))
            else:
                # pad-object
                obj_loc_fts = np.zeros((0,7), dtype=view_img_fts.dtype)
                nav_types.extend([2] * len(obj_loc_fts))
                batch_obj_img_fts.append(torch.zeros((0, 2048), dtype=batch_view_img_fts[0].dtype))
                batch_objids.append(np.array(['0'],dtype=object))
                batch_obj_lens.append(0)
                batch_loc_fts.append(torch.from_numpy(np.concatenate([view_loc_fts, obj_loc_fts], 0)))

            batch_nav_types.append(torch.LongTensor(nav_types))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        # object
        batch_obj_img_fts = pad_tensors(batch_obj_img_fts).cuda()
        batch_obj_lens = torch.LongTensor(batch_obj_lens).cuda()

        if not sum(have_objects):
            return {
                'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
                'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
                'cand_vpids': batch_cand_vpids,
            }
        else:
            return {
                'view_img_fts': batch_view_img_fts, 'obj_img_fts': batch_obj_img_fts,
                'loc_fts': batch_loc_fts, 'nav_types': batch_nav_types,
                'view_lens': batch_view_lens, 'obj_lens': batch_obj_lens,
                'cand_vpids': batch_cand_vpids, 'obj_ids': batch_objids,
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

    def nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts) + 1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        vp_masks = gen_seq_masks(view_lens + 1, max_len=vp_img_embeds.shape[-2])

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': vp_masks,
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': [[None] + x for x in cand_vpids],
        }

    def nav_vp_variable_object(self, obs, start_pos, pano_embeds, cand_vpids, view_lens, obj_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i in range(len(obs)):
            cur_cand_pos_fts = self.get_pos_fts(
                obs[i]['position'], [cc['position'] for cc in obs[i]['candidate']],
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
        vp_obj_masks = torch.cat([torch.zeros(batch_size, 1).bool().cuda(), nav_types == 2], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+obj_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_obj_masks': vp_obj_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)

        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes: # False
                    raise NotImplementedError
                    # if k == obs[i]['viewpoint']:
                    #     visited_vpids.append(k)
                    # else:
                    #     unvisited_vpids.append(k)
                else:
                    if 'eqa' in obs[i]['instr_id']:  # For EQA
                        unvisited_vpids.append(k)
                    else:
                        if gmap.graph.visited(k):
                            visited_vpids.append(k)
                        else:
                            unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:  # True
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                raise NotImplementedError
                # gmap_vpids = [None] + unvisited_vpids
                # gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )  # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i + 1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks,
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
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
                is_r2r = 'r2r' in ob['instr_id']
                if imitation_learning and is_r2r:
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
                                    pass
                                    # dist = - cal_dtw(
                                    #     self.shortest_distances[scan],
                                    #     sum(traj[i]['path'], []) + self.shortest_paths[scan][ob['viewpoint']][vpid][1:],
                                    #     ob['gt_path'],
                                    #     threshold=3.0
                                    # )['nDTW']
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
    # def make_equiv_action(self, a_t, obs, traj=None, env=None):
    #     """
    #     Interface between Panoramic view and Egocentric view
    #     It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
    #     """
    #     for i, ob in enumerate(obs):
    #         action = a_t[i]
    #         if action is not None:            # None is the <stop> action
    #             # traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
    #             traj[i]['path'].append([action])
    #             if len(traj[i]['path'][-1]) == 1:
    #                 prev_vp = traj[i]['path'][-2][-1]
    #             else:
    #                 prev_vp = traj[i]['path'][-1][-2]
    #             viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
    #             heading = (viewidx % 12) * math.radians(30)
    #             elevation = (viewidx // 12 - 1) * math.radians(30)
    #             env[i].sims[0].newEpisode([ob['scan']], [action], [heading], [elevation])

    def make_equiv_action(self, a_t, gmaps, obs, traj=None, env=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
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

        ml_loss, hist_infos = rollout(
            args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
            train_ml=train_ml, train_rl=train_rl, nav_agent=nav_agent, vln_model=vln_model,
            entropy_metric=entropy_metric,
        )

        if hist_infos is None:
            pass
        else:
            if train_ml is not None:
                iter_loss += ml_loss
                loss_metric.accumulate(ml_loss.item())

            #################### dagger training ####################
            feedback, train_ml, train_rl = 'sample', 1, False

            for b_i in range(len(batch_dict['env'])):
                scanIds = batch_dict['scanIds'][b_i]
                viewpointIds = batch_dict['viewpointIds'][b_i]
                headings = batch_dict['headings'][b_i]
                batch_dict['env'][b_i].newEpisodes(scanIds, viewpointIds, headings)
                observations = r2r_dataloader.dataset.get_obs(
                    items=[batch_dict['item'][b_i]],
                    env=batch_dict['env'][b_i],
                    data_type=batch_dict['data_type'][b_i]
                )[0]
                if batch_dict['data_type'][b_i] == 'eqa':
                    observations['answer'] = batch_dict['answer'][b_i]
                batch_dict['observations'][b_i] = observations

            sample_ml_loss, _ = rollout_sample(
                args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
                train_ml=train_ml, train_rl=train_rl, nav_agent=nav_agent, vln_model=vln_model,
                entropy_metric=entropy_metric, hist_infos=hist_infos
            )

            if train_ml is not None:
                iter_loss += sample_ml_loss
                loss_metric.accumulate(sample_ml_loss.item())

            if args.enable_language_model:
                raise NotImplementedError
            else:
                torch.nn.utils.clip_grad_norm_(vln_model.vln_bert.parameters(), 40.)
                vln_bert_optimizer.step()
                critic_optimizer.step()

        if args.rank == 0:
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
        is_val=False,  # in validation mode
):
    data_type = batch_dict['data_type']
    obs = batch_dict['observations']
    envs = batch_dict['env']

    if 'cvdn' in data_type or 'soon' in data_type:
        max_action_len = 20
    else:
        max_action_len = args.max_action_len  # 15

    nav_agent.update_scanvp_cands(obs)
    batch_size = len(obs)

    # build graph: keep the start viewpoint
    gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
    for i, ob in enumerate(obs):
        gmaps[i].update_graph(ob)

    # Record the navigation path
    traj = [{
        'instr_id': ob['instr_id'],
        'path': [[ob['viewpoint']]],
        'details': {},
        'eqa': defaultdict(list),
    } for ob in obs]

    # Initialization the tracking state
    ended = np.array([False] * batch_size)
    just_ended = np.array([False] * batch_size)

    instructions = nav_agent.language_variable(obs, data_type=batch_dict['data_type'])

    # Init the logs
    entropys = []
    ml_loss = 0.
    cnt_loss = 0.
    flag = False
    history = {}

    for t in range(max_action_len):
        if isinstance(vln_model.vln_bert, torch.nn.parallel.DistributedDataParallel):
            # multi-gpu
            if ended.all() or t == max_action_len - 1:
                flag = True
                context = nullcontext
            else:
                context = vln_model.vln_bert.no_sync
        else:
            # single-gpu
            if ended.all() or t == max_action_len - 1:
                break
                context = nullcontext
            else:
                context = nullcontext

        with context():
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = nav_agent.panorama_feature_variable_object(obs)
            pano_embeds, pano_masks = vln_model.vln_bert('panorama', pano_inputs)  # [B, 36, D=768], [B, 36,]
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)  # [B, D=768]

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    update_avg_pana_embeds = avg_pano_embeds[i].detach()  # update average features for gmap.
                    gmap.update_node_embed(i_vp, update_avg_pana_embeds, rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            update_pano_embeds = pano_embeds[i, j].detach()
                            gmap.update_node_embed(i_cand_vp, update_pano_embeds)

            # navigation policy
            nav_inputs = nav_agent.nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                nav_agent.nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': None,
                'txt_masks': None,
                'data_type': data_type,
                'step': t,
            })

            nav_outs = vln_model.vln_bert('navigation', nav_inputs)

            nav_vpids = nav_inputs['gmap_vpids']

            # Imitation Learning
            if train_ml is not None or feedback == 'gt':
                # Supervised training
                imitation_learning = feedback == 'teacher'
                nav_targets = nav_agent.teacher_action_r4r(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'],
                    imitation_learning=imitation_learning, t=t, traj=traj
                )
                if 'eqa' in data_type:
                    for idx in range(batch_size):
                        # if data_type[idx] == 'eqa' and nav_targets[idx] != nav_agent.args.ignoreid:
                        if data_type[idx] == 'eqa':
                            nav_targets[idx] = torch.tensor([obs[idx]['answer']], device=nav_targets.device)
                nav_outs[t].update({
                    'nav_targets': nav_targets
                })

            # Determinate the next navigation viewpoint
            if feedback == 'teacher':  # imitation learning
                a_t = nav_targets  # teacher forcing
            else:
                print(feedback)
                raise NotImplementedError

            if feedback == 'teacher' or feedback == 'sample':  # in training
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if data_type[i] == 'eqa':
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == args.max_action_len - 1):
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        cpu_a_t.append(nav_vpids[i][a_t[i]])

            history[t] = nav_outs

            # Make action and get the new state
            nav_agent.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)

            # get new observation and update graph
            new_obs = []
            for b_i in range(batch_size):
                if data_type[b_i] == 'eqa':
                    new_obs.append(obs[b_i])
                else:
                    new_obs.append(
                        r2r_dataloader.dataset.get_obs(
                            items=[batch_dict['item'][b_i]],
                            env=envs[b_i], data_type=data_type[b_i]
                        )[0]
                    )
            obs = new_obs

            nav_agent.update_scanvp_cands(obs)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            if flag:
                break

    # create instructions
    all_input_text = []
    for i, instr in enumerate(instructions):
        input_text = "{instruction}".format(instruction=instr)
        for t, item in history.items():
            input_text += item[t]['cand_text'][i]
        all_input_text.append(input_text)
    input_dict = {
        'all_input_text': all_input_text,
        'data_type': data_type,
        'history': history,
        'train_ml': train_ml
    }

    history_outs = vln_model.vln_bert('one_path', input_dict)
    truncation = history_outs['truncation']
    if truncation:
        # ml_loss, _ = rollout(
        #     args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
        #     train_ml=train_ml, train_rl=train_rl, nav_agent=nav_agent, vln_model=vln_model,
        #     entropy_metric=entropy_metric,
        # )
        return torch.tensor(0.).cuda(), None
    else:
        nav_probs = history_outs['nav_probs']
        nav_loss = history_outs['nav_loss']
        ml_loss = history_outs['ml_loss']
        cnt_loss = history_outs['cnt_loss']
        cnt_loss.backward()
        return ml_loss, {'hist_nav_probs': nav_probs, 'hist_nav_loss': nav_loss}


def rollout_sample(
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
        is_val=False,  # in validation mode
        hist_infos=None,
):
    data_type = batch_dict['data_type']
    obs = batch_dict['observations']
    envs = batch_dict['env']

    if 'cvdn' in data_type or 'soon' in data_type:
        max_action_len = 20
    else:
        max_action_len = args.max_action_len  # 15

    nav_agent.update_scanvp_cands(obs)
    batch_size = len(obs)

    # build graph: keep the start viewpoint
    gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
    for i, ob in enumerate(obs):
        gmaps[i].update_graph(ob)

    # Record the navigation path
    traj = [{
        'instr_id': ob['instr_id'],
        'path': [[ob['viewpoint']]],
        'details': {},
        'eqa': defaultdict(list),
    } for ob in obs]

    # Initialization the tracking state
    ended = np.array([False] * batch_size)
    just_ended = np.array([False] * batch_size)

    instructions = nav_agent.language_variable(obs, data_type=batch_dict['data_type'])

    # Init the logs
    entropys = []
    ml_loss = 0.
    cnt_loss = 0.
    flag = False
    history = {}

    for t in range(max_action_len):
        if isinstance(vln_model.vln_bert, torch.nn.parallel.DistributedDataParallel):
            # multi-gpu
            if ended.all() or t == max_action_len - 1:
                flag = True
                context = nullcontext
            else:
                context = vln_model.vln_bert.no_sync
        else:
            # single-gpu
            if ended.all() or t == max_action_len - 1:
                break
                context = nullcontext
            else:
                context = nullcontext

        with context():
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = nav_agent.panorama_feature_variable_object(obs)
            pano_embeds, pano_masks = vln_model.vln_bert('panorama', pano_inputs)  # [B, 36, D=768], [B, 36,]
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)  # [B, D=768]

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    update_avg_pana_embeds = avg_pano_embeds[i].detach()  # update average features for gmap.
                    gmap.update_node_embed(i_vp, update_avg_pana_embeds, rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            update_pano_embeds = pano_embeds[i, j].detach()
                            gmap.update_node_embed(i_cand_vp, update_pano_embeds)

            # navigation policy
            nav_inputs = nav_agent.nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                nav_agent.nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': None,
                'txt_masks': None,
                'data_type': data_type,
                'step': t,
            })

            nav_outs = vln_model.vln_bert('navigation', nav_inputs)

            nav_vpids = nav_inputs['gmap_vpids']

            # Imitation Learning
            if train_ml is not None or feedback == 'gt':
                # Supervised training
                imitation_learning = feedback == 'teacher'
                nav_targets = nav_agent.teacher_action_r4r(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'],
                    imitation_learning=imitation_learning, t=t, traj=traj
                )
                if 'eqa' in data_type:
                    for idx in range(batch_size):
                        # if data_type[idx] == 'eqa' and nav_targets[idx] != nav_agent.args.ignoreid:
                        if data_type[idx] == 'eqa':
                            nav_targets[idx] = torch.tensor([obs[idx]['answer']], device=nav_targets.device)
                nav_outs[t].update({
                    'nav_targets': nav_targets
                })

            # Determinate the next navigation viewpoint
            if feedback == 'sample':  # imitation learning
                fuse_embeds = nav_outs[t]['fuse_embeds']
                cand_masks = nav_outs[t]['cand_masks']
                gmap_visited_masks = nav_outs[t]['gmap_visited_masks']
                fuse_logits = torch.zeros((fuse_embeds.shape[0], fuse_embeds.shape[1])).to(
                    fuse_embeds.device)
                fuse_logits.masked_fill_(cand_masks.logical_not(), -float('inf'))
                fuse_logits.masked_fill_(gmap_visited_masks, -float('inf'))

                if t+1 > len(hist_infos['hist_nav_probs']):
                    for b_i in range(batch_size):
                        if nav_targets[b_i] == -100:
                            continue
                        else:
                            fuse_logits[b_i, nav_targets[b_i]] = t+1-len(hist_infos['hist_nav_probs'])
                else:
                    try:
                        for b_i in range(batch_size):
                            if nav_targets[b_i] == -100:
                                continue
                            elif t >= len(obs[b_i]['gt_path']):
                                fuse_logits[b_i, nav_targets[b_i]] = 1. / (hist_infos['hist_nav_loss'][t][b_i] + 1e-2)
                            elif obs[b_i]['viewpoint'] == obs[b_i]['gt_path'][t] and \
                                    fuse_logits.shape[-1] == hist_infos['hist_nav_probs'][t].shape[-1]:
                                fuse_logits[b_i] = hist_infos['hist_nav_probs'][t][b_i]
                            else:
                                fuse_logits[b_i, nav_targets[b_i]] = 1. / (hist_infos['hist_nav_loss'][t][b_i] + 1e-2)
                    except Exception as e:
                        print(e)
                nav_probs = torch.softmax(fuse_logits, 1)
                c = torch.distributions.Categorical(nav_probs.float())
                entropy_metric.accumulate(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            else:
                print(feedback)
                raise NotImplementedError

            if feedback == 'teacher' or feedback == 'sample':  # in training
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if data_type[i] == 'eqa':
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == args.max_action_len - 1):
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        cpu_a_t.append(nav_vpids[i][a_t[i]])

            history[t] = nav_outs

            # Make action and get the new state
            nav_agent.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)

            # get new observation and update graph
            new_obs = []
            for b_i in range(batch_size):
                if data_type[b_i] == 'eqa':
                    new_obs.append(obs[b_i])
                else:
                    new_obs.append(
                        r2r_dataloader.dataset.get_obs(
                            items=[batch_dict['item'][b_i]],
                            env=envs[b_i], data_type=data_type[b_i]
                        )[0]
                    )
            obs = new_obs

            nav_agent.update_scanvp_cands(obs)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            if flag:
                break

    # create instructions
    all_input_text = []
    for i, instr in enumerate(instructions):
        input_text = "{instruction}".format(instruction=instr)
        for t, item in history.items():
            input_text += item[t]['cand_text'][i]
        all_input_text.append(input_text)
    input_dict = {
        'all_input_text': all_input_text,
        'data_type': data_type,
        'history': history,
        'train_ml': train_ml
    }

    history_outs = vln_model.vln_bert('one_path', input_dict)
    truncation = history_outs['truncation']

    if truncation:
        # ml_loss, _ = rollout(
        #     args=args, r2r_dataloader=r2r_dataloader, batch_dict=batch_dict, feedback=feedback,
        #     train_ml=train_ml, train_rl=train_rl, nav_agent=nav_agent, vln_model=vln_model,
        #     entropy_metric=entropy_metric,
        # )
        return torch.tensor(0.).cuda(), None
    else:
        ml_loss = history_outs['ml_loss']
        cnt_loss = history_outs['cnt_loss']
        cnt_loss.backward()
        return ml_loss, None


def merge_dist_results(results):
    outs = []
    for res in results:
        outs.extend(res)
    return outs


def get_results(pred_results, detailed_output=False):
    pred_output = []
    for k, v in pred_results.items():
        if 'eqa' in k:
            pred_output.append(
                {
                    'instr_id': k,
                    'trajectory': v['path'],
                    'eqa': v['eqa']
                }
            )
        else:
            pred_output.append(
                {
                    'instr_id': k,
                    'trajectory': v['path']
                }
            )
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
        only_inference=False,
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

        ml_loss, traj = rollout_raw(
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

            if prefix == 'eqa':
                score_summary = r2r_dataloader.dataset.eval_eqa_metrics(
                    one_preds, logger=logger
                )
            else:
                score_summary, _ = r2r_dataloader.dataset.eval_metrics(
                    one_preds, logger=logger, data_type=prefix
                )

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

        if 'sr' in score_summary.keys():
            # select model by Success Rate
            if score_summary['sr'] >= best_val[args.val_split]['sr']:
                best_val[args.val_split]['spl'] = score_summary['spl']
                best_val[args.val_split]['sr'] = score_summary['sr']
                best_val[args.val_split]['state'] = 'Epoch %d %s' % (epoch, loss_str)

                save_ckpt_file = Path(args.run_name) / "best_{}".format(args.val_split)
                if not only_inference:
                    vln_model.save(epoch, str(save_ckpt_file),
                                   vln_bert_optimizer=vln_optimizer[0],
                                   critic_optimizer=vln_optimizer[1]
                                   )
        else:
            save_ckpt_file = Path(args.run_name) / "best_{}".format(args.val_split)
            if not only_inference:
                vln_model.save(epoch, str(save_ckpt_file),
                               vln_bert_optimizer=vln_optimizer[0],
                               critic_optimizer=vln_optimizer[1]
                               )


def rollout_raw(
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
        is_val=False,  # in validation mode
):
    data_type = batch_dict['data_type']
    obs = batch_dict['observations']
    envs = batch_dict['env']

    if 'cvdn' in data_type or 'soon' in data_type:
        max_action_len = 20
    else:
        max_action_len = args.max_action_len  # 15

    nav_agent.update_scanvp_cands(obs)
    batch_size = len(obs)

    # build graph: keep the start viewpoint
    gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
    for i, ob in enumerate(obs):
        gmaps[i].update_graph(ob)

    # Record the navigation path
    traj = [{
        'instr_id': ob['instr_id'],
        'path': [[ob['viewpoint']]],
        'details': {},
        'eqa': defaultdict(list),
    } for ob in obs]

    # Initialization the tracking state
    ended = np.array([False] * batch_size)
    just_ended = np.array([False] * batch_size)

    instructions = nav_agent.language_variable(obs, data_type=batch_dict['data_type'])

    history = []
    hist_vis = []
    for idx in range(len(instructions)):
        history.append("")
        hist_vis.append([])

    # Init the logs
    entropys = []
    ml_loss = 0.
    cnt_loss = 0.
    flag = False

    for t in range(max_action_len):
        if isinstance(vln_model.vln_bert, torch.nn.parallel.DistributedDataParallel):
            # multi-gpu
            if ended.all() or t == max_action_len - 1:
                flag = True
                context = nullcontext
            else:
                context = vln_model.vln_bert.no_sync
        else:
            # single-gpu
            if ended.all() or t == max_action_len - 1:
                break
                context = nullcontext
            else:
                context = nullcontext

        with context():
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = nav_agent.panorama_feature_variable_object(obs)
            pano_embeds, pano_masks = vln_model.vln_bert('panorama', pano_inputs)  # [B, 36, D=768], [B, 36,]
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)  # [B, D=768]

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    update_avg_pana_embeds = avg_pano_embeds[i].detach()  # update average features for gmap.
                    gmap.update_node_embed(i_vp, update_avg_pana_embeds, rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            update_pano_embeds = pano_embeds[i, j].detach()
                            gmap.update_node_embed(i_cand_vp, update_pano_embeds)

            # navigation policy
            nav_inputs = nav_agent.nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                nav_agent.nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': None,
                'txt_masks': None,
                'instruction': instructions,
                'history': history,
                'hist_vis': hist_vis,
                'data_type': data_type
            })

            nav_outs = vln_model.vln_bert('navigation_raw', nav_inputs)

            # dynamic fusion
            nav_logits = nav_outs['fuse_logits']
            nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }

            # Imitation Learning
            if train_ml is not None or feedback == 'gt':
                # Supervised training
                imitation_learning = feedback == 'teacher'
                nav_targets = nav_agent.teacher_action_r4r(
                    obs, nav_vpids, ended,
                    visited_masks=nav_inputs['gmap_visited_masks'],
                    imitation_learning=imitation_learning, t=t, traj=traj
                )
                if 'eqa' in data_type:
                    for idx in range(batch_size):
                        # if data_type[idx] == 'eqa' and nav_targets[idx] != nav_agent.args.ignoreid:
                        if data_type[idx] == 'eqa':
                            nav_targets[idx] = torch.tensor([obs[idx]['answer']], device=nav_targets.device)
                ############# Single-Step Loss #############
                cnt_loss += vln_model.criterion(nav_logits, nav_targets) * train_ml / batch_size
                ml_loss += cnt_loss.detach()
                ########### Single-Step Backward ###########
                if not is_val:
                    cnt_loss.backward()
                cnt_loss = 0.

            # Determinate the next navigation viewpoint
            if feedback == 'teacher':  # imitation learning
                a_t = nav_targets  # teacher forcing
            elif feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs.float())
                entropy_metric.accumulate(c.entropy().sum().item())  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
            elif feedback == 'argmax':
                _, a_t = nav_logits.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                if 'eqa' in data_type:
                    for idx in range(batch_size):
                        if data_type[idx] == 'eqa':
                            traj[idx]['eqa']['answer'].append(a_t.data.cpu())
                            traj[idx]['eqa']['scores'].append(nav_logits[idx].data.cpu())
                            traj[idx]['eqa']['gt'].append(obs[idx]['answer'])

                            _loss = vln_model.criterion(
                                nav_logits[idx:idx+1],
                                torch.tensor([obs[idx]['answer']]).cuda()).data.cpu()
                            traj[idx]['eqa']['loss'].append(
                                _loss
                            )
            else:
                print(feedback)
                raise NotImplementedError

            # Determine stop actions

            for idx in range(len(a_t)):
                if a_t[idx] == -100 or data_type[idx] == 'eqa':
                    continue
                history[idx] += '<hist>'
                hist_vis[idx].append(nav_outs['fuse_embeds'][idx][a_t[idx]])

            if feedback == 'teacher' or feedback == 'sample':  # in training
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if data_type[i] == 'eqa':
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == args.max_action_len - 1):
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        cpu_a_t.append(nav_vpids[i][a_t[i]])

            ##### instructions #####
            # TODO

            # Make action and get the new state
            nav_agent.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)

            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))

            # get new observation and update graph
            new_obs = []
            for b_i in range(batch_size):
                if data_type[b_i] == 'eqa':
                    new_obs.append(obs[b_i])
                else:
                    new_obs.append(
                        r2r_dataloader.dataset.get_obs(
                            items=[batch_dict['item'][b_i]],
                            env=envs[b_i], data_type=data_type[b_i]
                        )[0]
                    )
            obs = new_obs

            nav_agent.update_scanvp_cands(obs)

            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            if flag:
                break

    return ml_loss, traj
