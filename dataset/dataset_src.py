import copy
import random
import time
import numpy as np
import math
import json
import torch
import MatterSim
import networkx as nx
from torch.utils.data import DataLoader
import torchvision
from functools import partial
from tools.train import common_utils
from collections import defaultdict
from pathlib import Path
import torch.utils.data as torch_data
from torch.utils.data import DistributedSampler as _DistributedSampler
from dataset.process_multi_data import (
    load_r2r_data, load_soon_data,
    load_fr2r_data, load_reverie_data,
    generate_data_indexs,
    generate_graphs, load_nav_graphs,
    load_eqa_data, load_cvdn_data, load_cvdn_raw
)
from dataset.r2r_instr_src import InstructionPro
ERROR_MARGIN = 3.0
from duet.map_nav_src.r2r.eval_utils import cal_dtw, cal_cls


def new_simulator(connectivity_dir, scan_data_dir=None):
    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    # if scan_data_dir:
    #     sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim


def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]


class EnvBatch(object):
    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, batch_size=1):
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir: # None
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]

            if self.feat_db is None:
                feature = None
            else:
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class SrcDataset(torch_data.Dataset):
    def __init__(
            self,
            config,
            args,
            training=True,
            logger=None,
            image_processor=None,
            feat_db=None,
            tokenizer=None,
            test=False,
            angle_feat_size=4,
            split=None,
            obj_feat_db=None,
    ):
        self.config = config
        self.angle_feat_size = angle_feat_size

        if split is not None:
            self.split = split
        else:
            self.split = args.split

        self.logger = logger
        self.batch_size = args.batch_size
        self.training = training
        self.seed = args.seed
        self.feat_db = feat_db

        # object grounding ||| REVERIE
        if obj_feat_db is not None:
            assert isinstance(obj_feat_db, tuple)
            self.obj_db = obj_feat_db[0]
            self.obj2vps = obj_feat_db[1]  # {scan_objid: vp_list} (objects can be viewed at the viewpoints)
            self.soon_obj_db = obj_feat_db[2]
        else:
            self.obj_db = None
            self.obj2vps = None
            self.soon_obj_db = None
        self.multi_endpoints = False  # only for REVERIE dataset
        self.max_objects = 20

        _root_dir = Path(args.source_dir)

        # connectivity graph
        self.connectivity_dir = str(_root_dir / 'data/connectivity')

        # multi dataset
        msg = self._load_multi_data(config, _root_dir)

        if self.training:
            random.shuffle(self.alldata)

        self.buffered_state_dict = {}

        # simulator
        self.sim = new_simulator(self.connectivity_dir)

        # angle features
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        self._load_nav_graphs()

        if logger is not None:
            logger.info('[INFO] %s loaded with %d instructions, using splits: %s' % (
                self.__class__.__name__, len(self.alldata), self.split))
            logger.info(msg)

    def __len__(self):
        return len(self.alldata)

    def _load_multi_data(self, config, _root_dir, msg=''):
        self.data = dict()
        self.alldata = []
        self.gt_trajs = {}
        for source in config.SOURCE:
            if source == 'R2R':
                _anno_file = _root_dir / config.R2R.DIR / config.R2R.SPLIT[self.split]
                self.data['r2r'] = load_r2r_data(anno_file=_anno_file)

                # gt trajectories
                self.gt_trajs.update(
                    self.get_gt_trajs(self.data['r2r'])  # for evaluation
                )

                msg += '\n- Dataset: load {} R2R samples'.format(len(self.data['r2r']))
            elif source == 'SOON':
                _anno_file = _root_dir / config.SOON.DIR / config.SOON.SPLIT[self.split]
                self.data['soon'] = load_soon_data(anno_file=_anno_file)

                # gt trajectories
                self.gt_trajs.update(
                    self.get_gt_trajs_soon(self.data['soon'])  # for evaluation
                )

                msg += '\n- Dataset: load {} SOON samples'.format(len(self.data['soon']))

            elif source == 'REVERIE':
                _anno_file = _root_dir / config.REVERIE.DIR / config.REVERIE.SPLIT[self.split]
                self.data['reverie'] = load_reverie_data(anno_file=_anno_file)

                for item in self.data['reverie']:
                    if 'objId' in item and item['objId'] is not None:
                        item['end_vps'] = self.obj2vps['%s_%s' % (item['scan'], item['objId'])]

                # gt trajectories
                self.gt_trajs.update(
                    self.get_gt_trajs_reverie(self.data['reverie'])  # for evaluation
                )

                msg += '\n- Dataset: load {} REVERIE samples'.format(len(self.data['reverie']))

            elif source == 'CVDN':
                _anno_file = _root_dir / config.CVDN.DIR / config.CVDN.SPLIT[self.split]
                self.data['cvdn'] = load_cvdn_raw(anno_file=_anno_file)

                # gt trajectories
                self.gt_trajs.update(
                    self.get_gt_cvdn(self.data['cvdn'])  # for evaluation
                )

                msg += '\n- Dataset: load {} CVDN samples'.format(len(self.data['cvdn']))
            else:
                NotImplementedError

        for key, value in self.data.items():
            self.alldata += value
        msg += '\n- Dataset: load {} split: {} samples in total'.format(self.split, len(self.alldata))
        self.scans = set([x['scan'] for x in self.alldata])
        msg += '\n- Dataset: load {} split: {} scans in total'.format(self.split, len(self.scans))
        del self.data
        return msg

    def get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def get_gt_cvdn(self, data):
        gt_trajs = {}
        for x in data:
            gt_trajs[x['instr_id']] = x
        return gt_trajs

    def get_gt_trajs_reverie(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path'], x['objId']) \
                for x in data if 'objId' in x and x['objId'] is not None
        }
        return gt_trajs

    def get_gt_trajs_soon(self, data):
        # for evaluation
        gt_trajs = {
            x['instr_id']: copy.deepcopy(x) for x in data if 'bboxes' in x
        }
        return gt_trajs

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        # print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def get_obs(self, items, env, data_type=None):
        obs = []
        for i, (feature, state) in enumerate(env.getStates()):
            item = items[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)

            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            # if data_type == 'reverie' and self.obj_db is not None:
            #     # objects
            #     obj_img_fts, obj_ang_fts, obj_box_fts, obj_ids = self.obj_db.get_object_feature(
            #         state.scanId, state.location.viewpointId,
            #         state.heading, state.elevation, self.angle_feat_size,
            #         max_objects=self.max_objects # 20
            #     )

            # if (data_type == 'soon' or data_type == 'reverie') and self.soon_obj_db is not None:
            #     # objects
            if self.soon_obj_db is not None:
                obj_img_fts, obj_ang_fts, obj_box_fts, obj_directions, obj_ids = self.soon_obj_db.get_object_feature(
                    state.scanId, state.location.viewpointId,
                    state.heading, state.elevation, self.angle_feat_size,
                    max_objects=100
                )
            else:
                obj_img_fts, obj_ang_fts, obj_box_fts, obj_directions, obj_ids = [None]*5

            ob = {
                'instr_id': item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'candidate': candidate,
                'navigableLocations': state.navigableLocations,
                'instruction': item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path': item['path'],
                'path_id': item['path_id'],
            }

            # if data_type == 'reverie':
            #     ob.update({
            #         ### object grounding ###
            #         'obj_img_fts': obj_img_fts, # [1, 2048]
            #         'obj_ang_fts': obj_ang_fts, # [1,4]
            #         'obj_box_fts': obj_box_fts, # [1,3]
            #         'obj_ids': obj_ids, # [1,]
            #         'gt_end_vps': item.get('end_vps', []), # [multi vps]
            #         # 'gt_obj_id': item['objId'], #
            #     })
            #
            #     ### RL reward. The negative distance between the state and the final state
            #     ### There are multiple gt end viewpoints on REVERIE.
            #     # if ob['instr_id'] in self.gt_trajs:
            #     #     gt_objid = self.gt_trajs[ob['instr_id']][-1]
            #     #     min_dist = np.inf
            #     #     for vp in self.obj2vps['%s_%s' % (ob['scan'], str(gt_objid))]:
            #     #         try:
            #     #             min_dist = min(min_dist, self.shortest_distances[ob['scan']][ob['viewpoint']][vp])
            #     #         except:
            #     #             print(ob['scan'], ob['viewpoint'], vp)
            #     #             exit(0)
            #     #     ob['distance'] = min_dist
            #     # else:
            #     #     ob['distance'] = 0
            #
            # elif data_type == 'soon':
            #     ob.update({
            #         ### object grounding ###
            #         'obj_img_fts': obj_img_fts, # [n, 2048]
            #         'obj_ang_fts': obj_ang_fts, # [n,4]
            #         'obj_box_fts': obj_box_fts, # [n,3]
            #         'obj_ids': obj_ids, # [n,]
            #         'gt_end_vps': item.get('end_image_ids', []), # get multi-end-viewpoints
            #         # 'gt_obj_id': item['objId'], #
            #     })
            #
            # else:
            #     ob.update({
            #         ### object grounding ###
            #         'obj_img_fts': obj_img_fts,  # [1, 2048]
            #         'obj_ang_fts': obj_ang_fts,  # [1,4]
            #         'obj_box_fts': obj_box_fts,  # [1,3]
            #         'obj_ids': obj_ids,  # [1,]
            #     })

            # if ob['instr_id'] in self.gt_trajs:
            #     ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            # else:
            #     ob['distance'] = 0

            obs.append(ob)
        return obs

    def __getitem__(self, index):
        item = copy.deepcopy(self.alldata[index])
        data_type = item['data_type']
        scan = item['scan']
        instr_id = item['instr_id']

        # check length of instruction
        max_len = 128
        if len(item['instruction'].split()) > max_len:
            self.alldata[index]['instruction'] = " ".join(item['instruction'].split()[:max_len])
            item['instruction'] = " ".join(item['instruction'].split()[:max_len])

        if data_type == 'reverie':
            start_vp = item['path'][0]
            end_vp = item['path'][-1]
            # if self.multi_endpoints:
            #     end_vp = item['end_vps'][np.random.randint(len(item['end_vps']))]
            #     item = copy.deepcopy(self.alldata[index])
            #     item['path'] = self.shortest_paths[item['scan']][start_vp][end_vp]

        elif data_type == 'soon':
            if self.training:
                item['heading'] = np.random.rand() * np.pi * 2
                start_vp = item['path'][0]
                end_vp = item['path'][-1]

                # if self.multi_endpoints:
                #     end_vp = item['end_image_ids'][np.random.randint(len(item['end_image_ids']))]
                #     item = copy.deepcopy(self.alldata[index])
                #     item['path'] = self.shortest_paths[item['scan']][start_vp][end_vp]
            else:
                item['heading'] = 1.52
            item['elevation'] = 0

        elif data_type == 'cvdn':
            item['heading'] = item['start_pano']['heading']

        scanIds = [scan]
        viewpointIds = [item['path'][0]]
        headings = [item['heading']]

        env = EnvBatch(connectivity_dir=self.connectivity_dir, batch_size=1)
        env.newEpisodes(scanIds, viewpointIds, headings)
        observations = self.get_obs(items=[item], env=env, data_type=data_type)[0]

        data_dict = {
            'sample_idx': index,
            'instr_id': instr_id,
            'observations': observations,
            'env': env,
            'item': item,
            # 'instr': instr,
            'data_type': data_type,
        }

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['NotImplemented']:
                    ret[key] = torch.stack(val, 0)
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    ############### Nav Evaluation ###############
    def get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def eval_item(self, scan, pred_path, gt_path, gt_objid=None, is_soon=False, gt_item=None):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        start_vp = gt_path[0]
        goal_vp = gt_path[-1]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        if is_soon and gt_item is not None:
            gt_bboxes = gt_item['bboxes']
            # follow the original evaluation
            nearest_position = self.get_nearest(shortest_distances, goal_vp, path)
            if path[-1] in gt_bboxes:
                goal_vp = path[-1]  # update goal

            gt_lengths = shortest_distances[gt_path[0]][goal_vp]

            # navigation: success is navigation error < 3m
            scores['nav_error'] = shortest_distances[path[-1]][goal_vp]
            scores['oracle_error'] = shortest_distances[nearest_position][goal_vp]
            scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
            scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)
            scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        else:
            gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

            if gt_objid is not None:
                # navigation: success is to arrive to a viewpoint where the object is visible
                goal_viewpoints = set(self.obj2vps['%s_%s' % (scan, str(gt_objid))])
                assert len(goal_viewpoints) > 0, '%s_%s' % (scan, str(gt_objid))

                scores['success'] = float(path[-1] in goal_viewpoints)
                scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
                scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

            else:
                nearest_position = self.get_nearest(shortest_distances, gt_path[-1], path)

                scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
                scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

                scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
                scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
                scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

                scores.update(
                    cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
                )
                scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_cvdn(self, scan, path, gt_item):
        shortest_distances = self.shortest_distances[scan]

        start = gt_item['path'][0]
        assert start == path[0], 'Result trajectories should include the start position'
        goal = gt_item['path'][-1]
        planner_goal = gt_item['planner_path'][-1]  # for calculating oracle planner success (e.g., passed over desc goal?)
        final_position = path[0]  # 预测
        nearest_position = self.get_nearest(shortest_distances, goal, path)
        nearest_planner_position = self.get_nearest(shortest_distances, planner_goal, path)
        dist_to_end_start = None
        dist_to_end_end = None
        for end_pano in gt_item['end_panos']:
            d = shortest_distances[start][end_pano]
            if dist_to_end_start is None or d < dist_to_end_start:
                dist_to_end_start = d
            d = shortest_distances[final_position][end_pano]
            if dist_to_end_end is None or d < dist_to_end_end:
                dist_to_end_end = d
        scores = defaultdict(list)
        scores['nav_errors'].append(shortest_distances[final_position][goal])
        scores['oracle_errors'].append(shortest_distances[nearest_position][goal])
        scores['oracle_plan_errors'].append(shortest_distances[nearest_planner_position][planner_goal])
        scores['dist_to_end_reductions'].append(dist_to_end_start - dist_to_end_end)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev != curr:
                try:
                    self.graphs[gt_item['scan']][prev][curr]
                except KeyError as err:
                    print(err)
            distance += shortest_distances[prev][curr]
            prev = curr
        scores['trajectory_lengths'].append(distance)
        scores['shortest_path_lengths'].append(shortest_distances[start][goal])
        return scores

    def eval_dis_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self.get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds, logger=None, data_type=None):
        ''' Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]'''
        logger.info('eval %d predictions' % (len(preds))) if logger is not None else None

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']

            # TODO bugs, need to be fixed
            if instr_id not in self.gt_trajs.keys():
                print(instr_id)
                raise NotImplementedError
            # if data_type == 'reverie':
            #     scan, gt_traj, gt_objid = self.gt_trajs[instr_id]
            #     traj_scores = self.eval_item(scan, traj, gt_traj, gt_objid=gt_objid)
            # elif data_type == 'soon':
            #     gt_item = self.gt_trajs[instr_id]
            #     traj_scores = self.eval_item(gt_item['scan'], traj, gt_path=gt_item['path'], is_soon=True, gt_item=gt_item)
            # elif data_type == 'cvdn':
            #     traj = sum(item['trajectory'], [])
            #     scan, end_panos = self.gt_trajs[instr_id]
            #     traj_scores = self.eval_cvdn(scan, traj, end_panos)
            # else:
            #     scan, gt_traj = self.gt_trajs[instr_id]
            #     traj_scores = self.eval_item(scan, traj, gt_traj)

            if data_type == 'cvdn':
                traj = sum(item['trajectory'], [])
                gt_item = self.gt_trajs[instr_id]
                traj_scores = self.eval_cvdn(gt_item['scan'], traj, gt_item)
            elif data_type == 'soon':
                gt_item = self.gt_trajs[instr_id]
                traj_scores = self.eval_dis_item(gt_item['scan'], traj, gt_path=gt_item['path'])
            elif data_type == 'reverie':
                scan, gt_traj, gt_objid = self.gt_trajs[instr_id]
                traj_scores = self.eval_dis_item(scan, traj, gt_traj)
            else:
                scan, gt_traj = self.gt_trajs[instr_id]
                traj_scores = self.eval_dis_item(scan, traj, gt_traj)

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        if data_type == 'cvdn':
            num_successes = len([i for i in traj_scores['nav_errors'] if i < ERROR_MARGIN])
            oracle_successes = len([i for i in traj_scores['oracle_errors'] if i < ERROR_MARGIN])
            oracle_plan_successes = len([i for i in traj_scores['oracle_plan_errors'] if i < ERROR_MARGIN])

            avg_metrics = {
                'lengths': np.average(traj_scores['trajectory_lengths']),
                'nav_error': np.average(traj_scores['nav_errors']),
                'oracle_sr': float(oracle_successes) / float(len(traj_scores['oracle_errors'])),
                'sr': float(num_successes) / float(len(traj_scores['nav_errors'])),
                'spl': 0.0,
                'oracle path_success_rate': float(oracle_plan_successes) / float(
                    len(traj_scores['oracle_plan_errors'])),
                'dist_to_end_reduction': sum(traj_scores['dist_to_end_reductions']) / float(
                    len(traj_scores['dist_to_end_reductions']))
            }
        else:
            avg_metrics = {
                'action_steps': np.mean(metrics['action_steps']),
                'steps': np.mean(metrics['trajectory_steps']),
                'lengths': np.mean(metrics['trajectory_lengths']),
                'nav_error': np.mean(metrics['nav_error']),
                'oracle_error': np.mean(metrics['oracle_error']),
                'sr': np.mean(metrics['success']) * 100,
                'oracle_sr': np.mean(metrics['oracle_success']) * 100,
                'spl': np.mean(metrics['spl']) * 100,
                'nDTW': np.mean(metrics['nDTW']) * 100,
                'SDTW': np.mean(metrics['SDTW']) * 100,
                'CLS': np.mean(metrics['CLS']) * 100,
            }
        return avg_metrics, metrics


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset, batch_size, distributed, workers, training, seed=None):
    if distributed:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    shuffle = (sampler is None) and training

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=shuffle, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0,
        worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    dataloader.num_batches = len(dataloader)
    return dataset, dataloader, sampler