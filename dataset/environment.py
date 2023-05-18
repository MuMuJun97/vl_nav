import time
import numpy as np
import math
import json
import torchvision
from open_clip.transform import image_transform
import torch.utils.data as torch_data
from pathlib import Path
import random
import os
import math
from PIL import Image
import networkx as nx
import cv2
import copy
import pickle
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from tools.train import common_utils
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict
from dataset.process_multi_data import (
    load_r2r_data,load_soon_data,
    load_fr2r_data,load_reverie_data,
    generate_data_indexs,
    generate_graphs,load_nav_graphs,
    load_eqa_data, load_cvdn_data
)

def new_simulator(connectivity_dir):
    try:
        import MatterSim
    except Exception as e:
        print("[INFO] {}".format(e))
        print("[INFO] set env: export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH")
        exit()
    # Simulator image parameters
    WIDTH = 256
    HEIGHT = 256
    VFOV = 60

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setDiscretizedViewingAngles(True)  # Set increment/decrement to 30 degree. (otherwise by radians)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setBatchSize(1)
    sim.initialize()

    return sim

class ViewPoint(object):
    def __init__(
            self,
            viewpointId,
            ix=None,
            rel_distance=0.0,
            rel_elevation=0.0,
            rel_heading=0.0,
            x=0.0,
            y=0.0,
            z=0.0,
    ):
        self.ix = ix
        self.rel_distance = rel_distance
        self.rel_elevation = rel_elevation
        self.rel_heading = rel_heading
        self.viewpointId = viewpointId
        self.x = x
        self.y = y
        self.z = z

class SimState(object):
    def __init__(
            self,
            scanId,
            viewpointId,
            heading=0,
            elevation=0,
            step=0,
    ):
        self.depth = None
        self.rgb = None
        self.scanId = scanId
        self.elevation = elevation
        self.heading = heading
        self.location = ViewPoint(viewpointId)
        self.viewpointId = viewpointId
        self.step = step
        self.viewIndex, self.heading, self.elevation = self.setHeading(heading, elevation=elevation)
        self.navigableLocations = {
            viewpointId: ViewPoint(viewpointId),
        }

    def set_navigableLocations(self, viewpoint):
        self.navigableLocations.update(
            {
                viewpoint['viewpointId']: ViewPoint(viewpointId=viewpoint['viewpointId']),
            }
        )

    def setHeading(self, heading, headingCount=12, elevation=0):
        M_PI = math.pi
        elevationIncrement = M_PI / 6.0 # 30°
        state_heading = heading % (M_PI*2.0)
        while state_heading < 0.0:
            state_heading += math.pi*2.0
        # Snap heading to nearest discrete value
        headingIncrement = M_PI * 2.0 / headingCount
        heading_step = round(state_heading / headingIncrement)
        if heading_step == headingCount:
            heading_step = 0
        state_heading = heading_step * headingIncrement
        # Snap elevation to nearest discrete value (disregarding elevation limits)
        if elevation < -elevationIncrement/2.0:
            elevation = -elevationIncrement
            viewIndex = heading_step
        elif elevation > elevationIncrement/2.0:
            elevation = elevationIncrement
            viewIndex = heading_step + 2*headingCount
        else:
            elevation = 0.0
            viewIndex = heading_step + headingCount
        return viewIndex,state_heading,elevation


class Sim(object):
    def __init__(self,navigable_loc, headingCount=12):
        self.state = None
        self.headingCount = headingCount
        self.navigable_loc = navigable_loc

    def newEpisode(self, scanId, viewpointId, heading, elevation=0):
        self.state = SimState(scanId=scanId,viewpointId=viewpointId,heading=heading,elevation=elevation)
        nav_dict = self.navigable_loc[viewpointId]
        state_viewIndex = self.state.viewIndex - self.headingCount
        for k,v in nav_dict.items():
            self.state.set_navigableLocations(v)
            # if v['pointId'] == state_viewIndex:
            #     self.state.set_navigableLocations(v)

    def getState(self):
        return self.state

class EnvBatch(object):
    def __init__(self,shortest_paths,navigable_loc,img_dir,batch_size):
        self.shortest_paths = shortest_paths
        self.img_dir = img_dir
        self.navigable_loc = navigable_loc
        self.msims = []
        for i in range(batch_size):
            msim = Sim(navigable_loc)
            self.msims.append(msim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])
            self.msims[i].newEpisode(scanId,viewpointId,heading,0)

    def read_image(self, scan, viewpoint):
        img_file = self.img_dir / scan / '{}_{}.png'.format(scan,viewpoint)
        assert img_file.exists()
        panoramic_img = cv2.imread(str(img_file)) # BRG
        return panoramic_img

    def getStates(self):
        # states = []
        mstates = []
        # for i, sim in enumerate(self.sims):
        #     state = sim.getState()[0]
        #     panoramic_img = self.read_image(scan=state.scanId,viewpoint=state.location.viewpointId)
        #     states.append((panoramic_img,state))

        for i, msim in enumerate(self.msims):
            mstate = msim.getState()
            panoramic_img = self.read_image(scan=mstate.scanId,viewpoint=mstate.location.viewpointId)
            mstates.append((panoramic_img,mstate))

        return mstates

    def setHeading(self, heading, headingCount=12, elevation=0):
        M_PI = math.pi
        elevationIncrement = M_PI / 6.0 # 30°
        state_heading = heading % (M_PI*2.0)
        while state_heading < 0.0:
            state_heading += math.pi*2.0
        # Snap heading to nearest discrete value
        headingIncrement = M_PI * 2.0 / headingCount
        heading_step = round(state_heading / headingIncrement)
        if heading_step == headingCount:
            heading_step = 0
        state_heading = heading_step * headingIncrement
        # Snap elevation to nearest discrete value (disregarding elevation limits)
        if elevation < -elevationIncrement/2.0:
            elevation = -elevationIncrement
            viewIndex = heading_step
        elif elevation > elevationIncrement/2.0:
            elevation = elevationIncrement
            viewIndex = heading_step + 2*headingCount
        else:
            elevation = 0.0
            viewIndex = heading_step + headingCount
        return viewIndex,state_heading,elevation

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        # for i, (index, heading, elevation) in enumerate(actions):
        #     self.sims[i].makeAction([index], [heading], [elevation])
        raise NotImplementedError

    def shortest_path_action(self, state, goalViewpointId, gt_path):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            # current viewpoint is the goal location
            return state.location.viewpointId      # Just stop here
        if state.location.viewpointId in gt_path:
            nextViewpointId = gt_path[
                gt_path.index(state.location.viewpointId)+1
                ]
        else:
            path = self.shortest_paths[state.location.viewpointId][goalViewpointId]
            nextViewpointId = path[1]
        return nextViewpointId

    def get_obs(self,target_vp,gt_path):
        obs = []
        states = self.getStates()
        for i, (panoramic_img,state) in enumerate(states):

            # the agent may be located at other viewpoints and not on the gt trajectory
            # first: compute the next viewpoint
            teacher = self.shortest_path_action(state, target_vp, gt_path)

            candidate = copy.deepcopy(self.navigable_loc[state.location.viewpointId])
            candidate.pop(state.location.viewpointId) # remove self-viewpoint
            new_candidate = dict() # copy.deepcopy(candidate)
            view_id_candidate = dict()
            candidate_view_id = dict()

            for k,v in candidate.items():
                if view_id_candidate.get(v['pointId'],None) is None:
                    view_id_candidate[v['pointId']] = k
                    candidate_view_id[k] = v['pointId']
                    new_candidate[k] = candidate[k]
                else:
                    prev_vp = view_id_candidate[v['pointId']]
                    if prev_vp == teacher:
                        continue
                    elif k == teacher:
                        # remove previous viewpoint
                        view_id_candidate.pop(v['pointId'])
                        candidate_view_id.pop(prev_vp)
                        new_candidate.pop(prev_vp)
                        # add current new viewpoint
                        view_id_candidate[v['pointId']] = k
                        candidate_view_id[k] = v['pointId']
                        new_candidate[k] = candidate[k]
                    else:
                        # prev_vp and cur_vp are not GT-Label
                        if v['distance'] < candidate[prev_vp]['distance']:
                            view_id_candidate.pop(v['pointId'])
                            candidate_view_id.pop(prev_vp)
                            new_candidate.pop(prev_vp)

                            view_id_candidate[v['pointId']] = k
                            candidate_view_id[k] = v['pointId']
                            new_candidate[k] = candidate[k]
            del candidate
            ob = {
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'candidate': new_candidate,
                'teacher' : teacher, # self.shortest_path_action(state, target_vp, gt_path),
                'candidate_view_id': candidate_view_id,
                'view_id_candidate': view_id_candidate,
                'panoramic_img': panoramic_img,
                'navigableLocations' : state.navigableLocations,
            }
            obs.append(ob)
        return obs


class R2RNavBatch(object):
    def __init__(self,config,split='train',logger=None,batch_size=2,seed=0):
        self.config = config
        self.split = split
        self.logger = logger

        self.data_dir = Path(config.DATA_DIR).resolve()
        root_dir = Path(__file__).parent.parent.resolve()
        anno_file = root_dir/config.R2R.DIR/config.R2R.SPLIT[split]
        self.data = self.load_instr_datasets(anno_file=anno_file)
        self.scans = set([x['scan'] for x in self.data])

        ### mp3d navigable dict
        self.navigable_loc = self.get_navigable_Locations()

        self.gt_trajs = self._get_gt_trajs(self.data)  # for evaluation

        self.batch_size = batch_size

        connectivity_dir = str(root_dir/'data/connectivity')
        self.graphs = load_nav_graphs(connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        # self.shortest_distances = {}
        # for scan, G in self.graphs.items():  # compute all shortest paths
        #     self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        img_dir = Path(self.config.IMG_DIR).resolve()

        self.env = EnvBatch(
            navigable_loc=self.navigable_loc,
            img_dir=img_dir,
            batch_size=batch_size
        )

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        # index for iterate self.data
        self.ix = 0

        # self.sim = new_simulator(connectivity_dir)
        print('[INFO] %s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.split))

    def setHeading(self, heading, headingCount=12, elevation=0):
        M_PI = math.pi
        elevationIncrement = M_PI / 6.0 # 30°
        state_heading = heading % (M_PI*2.0)
        while state_heading < 0.0:
            state_heading += math.pi*2.0
        # Snap heading to nearest discrete value
        headingIncrement = M_PI * 2.0 / headingCount
        heading_step = round(state_heading / headingIncrement)
        if heading_step == headingCount:
            heading_step = 0
        state_heading = heading_step * headingIncrement
        # Snap elevation to nearest discrete value (disregarding elevation limits)
        if elevation < -elevationIncrement/2.0:
            elevation = -elevationIncrement
            viewIndex = heading_step
        elif elevation > elevationIncrement/2.0:
            elevation = elevationIncrement
            viewIndex = heading_step + 2*headingCount
        else:
            elevation = 0.0
            viewIndex = heading_step + headingCount
        return viewIndex,state_heading,elevation

    def size(self):
        return len(self.data)

    def get_navigable_Locations(self):
        """
        :return:
         exp: ['2t7WUuJeko7']['1e6b606b44df4a6086c0f97e826d4d15'] (current viewpoint Id)
            {
             '1e3a672fa1d24d668866455162e5b58a': (navigable adjacent viewpoint Id)
             {
               'heading': loc_heading,
               'elevation': loc_elevation,
               "normalized_heading": state.heading + loc.rel_heading,
               "normalized_elevation": state.elevation + loc.rel_elevation,
               'scanId': scan_id, # sets which scene is used, e.g. "2t7WUuJeko7"
               'viewpointId': loc.viewpointId,  # sets the adjacent viewpoint location,
               'pointId': ix, # 当前viewpoint的第ix-th个view指向loc.viewpointId [0-11]
               'distance': distance,
               'idx': j + 1, # adjacent index
               'position': (loc.x, loc.y, loc.z),
             }
            }
        """
        mp3d_nav_file = Path(__file__).parent.parent.resolve() / self.config.MP3D_NAV
        with open(str(mp3d_nav_file),"rb") as f:
            res_dict = pickle.load(f)
        return res_dict

    def load_instr_datasets(self,anno_file):
        assert anno_file.exists()
        with open(str(anno_file)) as f:
            data = json.load(f)
        new_data = []
        for i, item in enumerate(data):
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                del new_item['instructions']
                del new_item['instr_encodings']
                new_data.append(new_item)
        return new_data

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def get_obs(self):
        obs = []
        states = self.env.getStates()
        for i, (panoramic_img,state) in enumerate(states):
            item = self.batch[i]
            base_view_id = state.viewIndex

            candidate = copy.deepcopy(self.navigable_loc[state.scanId][state.location.viewpointId])
            candidate.pop(state.location.viewpointId)
            view_id_candidate = {v['pointId']:k for k,v in candidate.items()}
            candidate_view_id = {k:v['pointId'] for k,v in candidate.items()}

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'candidate': candidate,
                'candidate_view_id': candidate_view_id,
                'view_id_candidate': view_id_candidate,
                'panoramic_img': panoramic_img,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'gt_path' : item['path'],
                'path_id' : item['path_id']
            }

            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE.
            # if ob['instr_id'] in self.gt_trajs:
            #     ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            # else:
            #     ob['distance'] = 0

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]

        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self.get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self.get_obs()

    def make_equiv_action(self, pred_viewpoint, obs, traj=None):
        for i, ob in enumerate(obs):
            action = pred_viewpoint[i]
            if action is not None:  # None is the <stop> action
                traj[i]['path'].append(action)
                prev_vp = traj[i]['path'][-2]
                viewidx = self.navigable_loc[ob['scan']][prev_vp][action]['pointId']
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.msims[i].newEpisode(ob['scan'], action, heading, elevation)

    # TODO ############### Nav Evaluation ###############
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

    def eval_item(self, scan, pred_path, gt_path, ERROR_MARGIN = 3.0):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = pred_path
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

        # TODO dtw
        # TODO CLS

        return scores


############################### Dataset ###############################
class R2RDataset(torch_data.Dataset):
    def __init__(self, config, args, training=True, logger=None, image_processor=None, tokenizer=None, test=False):
        self.config = config
        self.split = args.split
        self.logger = logger
        self.batch_size = args.batch_size
        self.training = training
        self.seed = args.seed

        _root_dir = Path(__file__).parent.parent.resolve()

        # connectivity graph
        connectivity_dir = str(_root_dir / 'data/connectivity')
        graph_dict_file = _root_dir / 'build/R2R/R2R_{}_graph_dict.pkl'.format(self.split)
        with open(str(_root_dir/"data/connectivity/scans.txt")) as f:
            mp3d_scans = f.readlines()
        mp3d_scans = [s.strip() for s in mp3d_scans]

        _, self.shortest_paths = generate_graphs(
            graph_dict_file=graph_dict_file,
            rank=args.rank,
            logger=logger,
            connectivity_dir=connectivity_dir,
            scans=mp3d_scans,
        )

        # multiple dataset
        self.data = dict()
        self.alldata = []
        self.all_index = dict()
        msg = ''
        for source in config.SOURCE:
            if source == 'R2R':
                _anno_file = _root_dir/config.R2R.DIR/config.R2R.SPLIT[self.split]
                self.data['r2r'] = load_r2r_data(anno_file=_anno_file)
                msg += '\n- Dataset: load {} R2R samples'.format(len(self.data['r2r']))
            elif source == 'REVERIE':
                _anno_file = _root_dir/config.REVERIE.DIR/config.REVERIE.SPLIT[self.split]
                self.data['reverie'] = load_reverie_data(anno_file=_anno_file)
                msg += '\n- Dataset: load {} REVERIE samples'.format(len(self.data['reverie']))
            elif source == 'SOON':
                _anno_file = _root_dir/config.SOON.DIR/config.SOON.SPLIT[self.split]
                self.data['soon'] = load_soon_data(anno_file=_anno_file)
                msg += '\n- Dataset: load {} SOON samples'.format(len(self.data['soon']))
            elif source == "FR2R":
                _anno_file = _root_dir/config.FR2R.DIR/config.FR2R.SPLIT[self.split]
                self.data['fr2r'] = load_fr2r_data(anno_file=_anno_file)
                msg += '\n- Dataset: load {} Fine-grained R2R samples'.format(len(self.data['fr2r']))
            elif source == "EQA":
                _anno_file = _root_dir/config.EQA.DIR
                eqa_split = 'train'
                if 'val' in self.split:
                    eqa_split = 'val'
                self.data['eqa'] = load_eqa_data(anno_file=_anno_file,split=eqa_split)
                msg += '\n- Dataset: load {} Embodied QA samples'.format(len(self.data['eqa']))
            elif source == "CVDN":
                _anno_file = _root_dir/config.CVDN.DIR/config.CVDN.SPLIT[self.split]
                self.data['cvdn'] = load_cvdn_data(anno_file=_anno_file, shortest_paths=self.shortest_paths)
                msg += '\n- Dataset: load {} CVDN samples'.format(len(self.data['cvdn']))
            else:
                NotImplementedError

        self.alldata, self.all_index = generate_data_indexs(data=self.data)
        msg += '\n- Dataset: load {} split: {} samples in total'.format(self.split, len(self.alldata))
        del self.data

        self.scans = set([x['scan'] for x in self.alldata])
        msg += '\n- Dataset: load {} split: {} scans in total'.format(self.split, len(self.scans))

        # MP3D Connectivity Graph
        self.navigable_loc = self.get_navigable_Locations()

        # Image Dir
        self.img_dir = Path(self.config.IMG_DIR).resolve()
        self.image_preprocess = image_transform(
            self.config.vision_encoder.image_size,
            is_train=False,
            mean=None,
            std=None
        )

        if args.shuffle:
            random.seed(self.seed)
            random.shuffle(self.alldata) if self.training else None

        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.input_text = []
        self.vis_infos = []
        self.last_vp = []
        for index in range(len(self.alldata)):
            if test:
                input_text, vis_infos, last_vp = self.pre_process_test(index)
            else:
                input_text, vis_infos, last_vp = self.pre_process(index)
            self.input_text.append(input_text)
            self.vis_infos.append(vis_infos)
            self.last_vp.append(last_vp)

        if logger is not None:
            logger.info('[INFO] %s loaded with %d instructions, using splits: %s' % (
                self.__class__.__name__, len(self.alldata), self.split))
            logger.info(msg)

    def get_navigable_Locations(self):
        """
        :return:
         exp: ['2t7WUuJeko7']['1e6b606b44df4a6086c0f97e826d4d15'] (current viewpoint Id)
            {
             '1e3a672fa1d24d668866455162e5b58a': (navigable adjacent viewpoint Id)
             {
               'heading': loc_heading,
               'elevation': loc_elevation,
               "normalized_heading": state.heading + loc.rel_heading,
               "normalized_elevation": state.elevation + loc.rel_elevation,
               'scanId': scan_id, # sets which scene is used, e.g. "2t7WUuJeko7"
               'viewpointId': loc.viewpointId,  # sets the adjacent viewpoint location,
               'pointId': ix, # 当前viewpoint的第ix-th个view指向loc.viewpointId [0-11]
               'distance': distance,
               'idx': j + 1, # adjacent index
               'position': (loc.x, loc.y, loc.z),
             }
            }
        """
        mp3d_nav_file = Path(__file__).parent.parent.resolve() / self.config.MP3D_NAV
        with open(str(mp3d_nav_file),"rb") as f:
            res_dict = pickle.load(f)
        return res_dict

    def get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def __len__(self):
        return len(self.alldata)

    def load_images(self, scan, vp, valid_ids=None):
        img_file = self.img_dir / scan / '{}_{}.png'.format(scan, vp)
        assert img_file.exists()
        panoramic_img = cv2.imread(str(img_file))  # BRG
        img_12 = np.hsplit(panoramic_img, 12)
        if valid_ids is not None:
            img_12 = [img_12[i] for i in valid_ids]
        images = [
            self.image_processor(
                Image.fromarray(
                    s[:, :, ::-1]  # BRG2RGB
                )
            ).unsqueeze(0)
            for s in img_12
        ]
        images = torch.cat(images, dim=0)  # [12,3,224,224]
        if self.training:
            # apply random horizontal flip and color jitter
            images = torchvision.transforms.RandomHorizontalFlip(p=0.5)(images)
            images = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(images)

        return images

    def setHeading(self, heading, headingCount=12, elevation=0):
        M_PI = math.pi
        elevationIncrement = M_PI / 6.0 # 30°
        state_heading = heading % (M_PI*2.0)
        while state_heading < 0.0:
            state_heading += math.pi*2.0
        # Snap heading to nearest discrete value
        headingIncrement = M_PI * 2.0 / headingCount
        heading_step = round(state_heading / headingIncrement)
        if heading_step == headingCount:
            heading_step = 0
        state_heading = heading_step * headingIncrement
        # Snap elevation to nearest discrete value (disregarding elevation limits)
        if elevation < -elevationIncrement/2.0:
            elevation = -elevationIncrement
            viewIndex = heading_step
        elif elevation > elevationIncrement/2.0:
            elevation = elevationIncrement
            viewIndex = heading_step + 2*headingCount
        else:
            elevation = 0.0
            viewIndex = heading_step + headingCount
        return viewIndex,state_heading,elevation

    def compute_angle_features(self, candidates: dict, heading: float):
        """
        Args:
            candidates: candidate viewpoints, dict{}
            heading: float, current viewpoint->heading

        Returns:

        """
        valid_view = dict()
        view_index, state_heading, _ = self.setHeading(heading)
        base_heading = state_heading
        cur_vp = candidates[list(candidates.keys())[0]]

        for vp, candidate in candidates.items():
            if vp == cur_vp['viewpointId']:
                continue
            if valid_view.get(candidate['pointId'], None) is None:
                valid_view[candidate['pointId']] = dict()

            # "normalized_heading": state.heading + loc.rel_heading
            normalized_heading = candidate['normalized_heading']

            # adj_dict[loc.viewpointId]['heading']: state.heading - base_heading + loc.rel_heading
            adj_heading = normalized_heading - base_heading
            adj_angle_feature = np.array(
                [math.sin(adj_heading), math.cos(adj_heading)],
                dtype=np.float32
            )
            valid_view[candidate['pointId']].update({
                'angle_feats': adj_angle_feature,
                'viewpointId': candidate['viewpointId'],
            })
        return valid_view

    def get_vis(self, vis_infos):
        input_image = []
        input_angle_feats = []
        for (scan, vp, valid_view) in vis_infos:
            images = self.load_images(scan, vp, valid_ids=list(valid_view.keys()))
            input_image.append(images)

            per_angle_feats = []
            for k, v in valid_view.items():
                per_angle_feats.append(v['angle_feats'])
            input_angle_feats.append(torch.from_numpy(
                np.stack(per_angle_feats)
            ))
        input_image = torch.cat(input_image, dim=0)
        input_angle_feats = torch.cat(input_angle_feats, dim=0)
        return input_image, input_angle_feats

    def load_multi_step_data(self, paths, texts, instruction, navigable_dict, scan,
                             tokenizer=None, item=None):
        """
        Args:
            paths: [start_viewpoint, ..., end_viewpoint]
            instruction: "walk to ..." navigation instruction
            navigable_dict: dict()
                navigable_dict[viewpoint][next_viewpoint]:{
                    'pointId': next view id,
                }
            scan:
            tokenizer:

        Multi-Step Dialog Settings:
            Task-Description: xxx
            #Step 1, Environment: <image0>...<image11>
            #Action:<walkto?>
            #Step 2, Environment: <image0>...<image11>
            #Action:<walkto?>
            ...
        Returns:

        """
        prompt = "System: {task_description}" \
                 "\nCommander: {instruction}" \
                 "{history}"
        task_description = "You are a mobile agent in an indoor building." \
                        #    "I will provide 12 images of the environment in different directions from the current location." \
                        #    "Given an input instruction, you need to move to the next location " \
                        #    "based on the current environment and historical trajectory." \
                        #    "Please use <walkto0>,<walkto1>,<walkto2>,<walkto3>,<walkto4>,<walkto5>," \
                        #    "<walkto6>,<walkto7>,<walkto8>,<walkto9>,<walkto10>,<walkto11> to move " \
                        #    "or <stop> to stop."

        trajs = paths # T+1 steps
        history_text = ""
        vis_infos = []
        heading = item['heading'] if item.get('heading', None) is not None else 0.

        trajs_len = len(trajs)
        for t, vp in enumerate(trajs):
            if vp is None:
                if t in texts:
                    for idx, text in enumerate(texts[t]):
                        if idx % 2 == 0:
                            history_text += "\nAgent: <s> {} </s>".format(text)
                        else:
                            history_text += "\nCommander: {}".format(text)
                break

            # Vision:
            valid_view = self.compute_angle_features(
                candidates=navigable_dict[vp],
                heading=heading,
            )

            history_text += "\nEnvironment: " + "".join([
                '<image{}>'.format(x, x) if x in valid_view.keys() else ''
                for x in range(12)
            ])

            vis_infos.append((scan, vp, valid_view))

            # Text:
            if t in texts:
                for idx, text in enumerate(texts[t]):
                    if idx % 2 == 0:
                        history_text += "\nAgent: <s> {} </s>".format(text)
                    else:
                        history_text += "\nCommander: {}".format(text)

            # Action:
            if t != trajs_len - 1:
                next_vp = trajs[t + 1]
                if next_vp is None:
                    history_text += "\nAgent: <s><stop></s>"
                else:
                    next_view_id = navigable_dict[vp][next_vp]['pointId']
                    assert next_view_id in list(valid_view.keys())
                    history_text += "\nAgent: <s><walkto{}></s>".format(next_view_id)
                    heading = (next_view_id % 12) * math.radians(30)

        input_text = prompt.format(
            task_description=task_description,
            instruction=instruction,
            history=history_text,
        )

        return input_text, vis_infos

    def get_instruction(self, item):
        data_type = item['data_type']
        if data_type == 'r2r':
            return 'Travel following the instruction, you can not ask for help. Instruction: ' \
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

    def get_gt_action(self, scan, loc_a, loc_b):
        #TODO
        if loc_a == loc_b:
            return None
        path = self.shortest_paths[item['scan']][loc_a][loc_b]
        import pdb;pdb.set_trace()

    def pre_process_test(self, index):
        item = self.alldata[index]
        scan = item['scan']
        data_type = item['data_type']
        texts = {}
        instr_id = item['instr_id']

        instruction = self.get_instruction(item)
        if data_type == 'r2r':
            paths = item['path'][:1]
     
        elif data_type == 'soon':
            paths = item['path'][:1]

        elif data_type == 'reverie':
            paths = item['path'][:1]

        elif data_type == 'eqa':
            paths = item['path'][:1]

        elif data_type == 'cvdn':
            paths = item['path'][:item['history_id']+1]
            texts = item['texts']

        input_text, vis_infos \
            = self.load_multi_step_data(
                paths=paths,
                texts=texts,
                instruction=instruction,
                navigable_dict=self.navigable_loc[scan],
                scan=scan,
                item=item,
            )
        input_text += "\nAgent: <s>"
        return input_text, vis_infos, paths[-1]      

    def make_equiv_action(self, scan, vp, action):
        if action == '<stop>':
            return None
        else:
            action_id = eval(action[7:-1]) # <walkto12>
            all_adj = self.navigable_loc[scan][vp]
            for k in all_adj:
                if all_adj[k]['pointId'] == action_id:
                    next_scan = all_adj[k]
            assert next_scan is not None
            vp = next_scan['viewpointId']
            heading = next_scan['heading']
            valid_view = self.compute_angle_features(
                candidates=self.navigable_loc[scan][vp],
                heading=heading,
            )
            text = action + '</s>'
            text += "\nEnvironment: " + "".join([
                '<image{}>'.format(x, x) if x in valid_view.keys() else ''
                for x in range(12)
            ])
            text += "\nAgent: <s>"
            vis_infos=[(scan, vp, valid_view)]
            input_image, input_angle_feats = self.get_vis(vis_infos)
            input_image = [input_image]
            input_angle_feats = [input_angle_feats]

            return {
                'batch_size': 1,
                'scan': scan,
                'vp': vp,
                'input_text': text,
                'vis_infos': vis_infos,
                'input_image': input_image,
                'input_angle_feats': input_angle_feats,
            }

    def pre_process(self, index):
        item = self.alldata[index]
        scan = item['scan']
        data_type = item['data_type']
        texts = {}
        instr_id = item['instr_id']
        instruction = self.get_instruction(item)
        paths = item['path'] + [None]

        if data_type == 'eqa':
            # TODO validate correctness
            texts = {k+1:v for k,v in item['texts'].items()}

        if data_type == 'cvdn':
            texts = item['texts']

        input_text, vis_infos \
            = self.load_multi_step_data(
                paths=paths,
                texts=texts,
                instruction=instruction,
                navigable_dict=self.navigable_loc[scan],
                scan=scan,
                item=item,
            )
        return input_text, vis_infos, paths[-1]

    def update_data(self, index, input_text, vis_infos):
        pass

    def __getitem__(self, index):
        item = self.alldata[index]
        scan = item['scan']
        data_type = item['data_type']
        instr_id = item['instr_id']
        instruction = self.get_instruction(item)
        gt_paths = item['path']

        input_text, vis_infos, last_vp = self.input_text[index], self.vis_infos[index], self.last_vp[index]

        input_image, input_angle_feats = self.get_vis(vis_infos)

        return {
            'data_type': data_type, # 'r2r' 'soon' ...
            'sample_idx': index,
            'instr_id': instr_id,
            'scan': scan,
            'vp': last_vp,
            'gt_paths': gt_paths,
            'instruction': instruction,
            'input_text': input_text,
            'vis_infos': vis_infos,
            'input_image': input_image,
            'input_angle_feats': input_angle_feats,
        }
    
    def __getrawitem__(self, index):
        return self.alldata[index]



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


def build_dataloader(args, dataset, seed=None):
    batch_size = args.batch_size
    dist = args.distributed
    workers = args.workers
    training = False if args.split != 'train' else True
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if args.shuffle:
        shuffle = (sampler is None) and training
    else:
        shuffle = False

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=shuffle, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    dataloader.num_batches = len(dataloader)
    return dataset, dataloader, sampler
