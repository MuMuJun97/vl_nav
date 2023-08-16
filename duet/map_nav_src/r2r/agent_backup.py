import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from networks.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts
from networks.model import VLNBert, Critic
from networks.ops import pad_tensors_wgrad
from contextlib import nullcontext

class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}
    
    def get_instruction(self, item):
        data_type = 'r2r' #TODO
        if data_type == 'r2r':
            return 'Instruction: ' \
                + item['instruction']
        elif data_type == 'soon':
            return 'Find the described target, you can not ask for help. Target: ' \
                + item['instruction']['instruction']
        elif data_type == 'reverie':
            return 'Go to the location to complete the given task, you can not ask for help. Task: ' \
                + item['instruction']
        elif data_type == 'eqa':
            return 'Explore the scene and answer the question, you can not ask for help. Question: ' \
                          + item['instruction']
        elif data_type == 'cvdn':
            return 'Find the described target, you can ask for help. Target: ' \
                + item['instruction']

    def _language_variable(self, obs):
        raw_instruction = [self.get_instruction(ob) for ob in obs]
        return raw_instruction

    def _panorama_feature_variable(self, obs):
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
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
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

    def _nav_vp_variable(self, obs, start_pos, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i in range(len(obs)):
            cand_pos = [cc['position']for cc in obs[i]['candidate']]
            cur_cand_pos_fts = self.get_pos_fts(
                obs[i]['position'], cand_pos, 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = self.get_pos_fts(
                obs[i]['position'], [start_pos[i]], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 4+4+3), dtype=np.float32)
            vp_pos_fts[:, :4] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 4:8] = cur_cand_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 8:] = np.array(cand_pos)

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

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r(
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
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append([action])
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True, do_backward=False):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]

        start_pos = [ob['position'] for ob in obs]
        
        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        instruction = language_inputs
        history = []
        hist_vis = []
        for idx in range(len(instruction)):
            history.append("")
            hist_vis.append([])
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.  
        flag = False
        for t in range(self.args.max_action_len):
            if ended.all() or t == self.args.max_action_len-1:
                flag = True
                context = nullcontext
            else:
                context = self.vln_bert.no_sync

            with context():

                pano_inputs = self._panorama_feature_variable(obs)
                pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)

                # avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                #                   torch.sum(pano_masks, 1, keepdim=True)

                # navigation policy
                nav_inputs = self._nav_vp_variable(
                    obs, start_pos, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['nav_types'],
                )

                nav_inputs['instruction'] = instruction
                nav_inputs['history'] = history
                nav_inputs['hist_vis'] = hist_vis

                # import pdb;pdb.set_trace()
                nav_outs = self.vln_bert('navigation', nav_inputs)

                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
                nav_probs = torch.softmax(nav_logits, 1)

                for idx in range(len(instruction)):
                    if ended[idx]:
                        continue
                    history[idx] += '<'
                    hist_vis[idx].append(nav_outs['vp_embeds'][idx])

                if train_ml is not None:
                    # Supervised training
                    if self.args.dataset == 'r2r':
                        nav_targets = self._teacher_action_r4r(
                            obs, nav_vpids, ended, 
                            visited_masks=None,
                            imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                        )
                    elif self.args.dataset == 'r4r':
                        nav_targets = self._teacher_action_r4r(
                            obs, nav_vpids, ended, 
                            visited_masks=None,
                            imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                        )
                    # print(t, nav_logits, nav_targets)
                    cnt_loss = self.criterion(nav_logits, nav_targets) * train_ml / batch_size
                    ml_loss += cnt_loss.detach()
                    cnt_loss.backward()
                    cnt_loss = 0.
                           
                # Determinate the next navigation viewpoint
                if self.feedback == 'teacher':
                    a_t = nav_targets                 # teacher forcing
                elif self.feedback == 'argmax':
                    _, a_t = nav_logits.max(1)        # student forcing - argmax
                    a_t = a_t.detach() 
                elif self.feedback == 'sample':
                    c = torch.distributions.Categorical(nav_probs.float())
                    self.logs['entropy'].append(c.entropy().sum().item())            # For log
                    entropys.append(c.entropy())                                     # For optimization
                    a_t = c.sample().detach() 
                elif self.feedback == 'expl_sample':
                    _, a_t = nav_probs.max(1)
                    rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                    if self.args.fusion == 'local':
                        cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                    else:
                        cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                    for i in range(batch_size):
                        if rand_explores[i]:
                            cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                            a_t[i] = np.random.choice(cand_a_t)
                else:
                    print(self.feedback)
                    sys.exit('Invalid feedback option')

                # self.vln_bert('add_history', {'vis':nav_outs['vp_embeds'], 'action': a_t})

                # Determine stop actions
                if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                    # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                    a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
                else:
                    a_t_stop = a_t == 0

                # Prepare environment action
                cpu_a_t = []  
                for i in range(batch_size):
                    if a_t_stop[i] or ended[i] or (t == self.args.max_action_len - 1):
                        cpu_a_t.append(None)
                        just_ended[i] = True
                    else:
                        cpu_a_t.append(nav_vpids[i][a_t[i]])   

                # Make action and get the new state
                self.make_equiv_action(cpu_a_t, obs, traj)

                # new observation and update graph
                obs = self.env._get_obs()
                self._update_scanvp_cands(obs)

                ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

                # Early exit if all ended
                # if ended.all():
                #     break
                if flag:
                    break

        if train_ml is not None:
            self.logs['IL_loss'].append(ml_loss.item())
        return traj
