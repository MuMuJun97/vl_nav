import time
import numpy as np
import math
import json
from open_clip.transform import image_transform
import torch.utils.data as torch_data
from pathlib import Path
import random
import os
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

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


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
        self.viewIndex,self.heading,self.elevation = self.setHeading(heading,elevation=elevation)
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
    def __init__(self, config, args, training=True, logger=None):
        self.config = config
        self.split = args.split
        self.logger = logger
        self.batch_size = args.batch_size
        self.training = training
        self.seed = args.seed

        #### R2R Data ####
        self.data_dir = Path(config.DATA_DIR).resolve()
        root_dir = Path(__file__).parent.parent.resolve()
        anno_file = root_dir/config.R2R.DIR/config.R2R.SPLIT[self.split]

        self.data = self.load_instr_datasets(anno_file=anno_file)
        self.scans = set([x['scan'] for x in self.data])
        self.gt_trajs = self.get_gt_trajs(self.data)  # for evaluation

        #### MP3D Connectivity Graph ####
        self.navigable_loc = self.get_navigable_Locations()

        #### connectivity graph ####
        connectivity_dir = str(root_dir / 'data/connectivity')
        graph_dict_file = root_dir / 'build/R2R/R2R_{}_graph_dict.pkl'.format(self.split)
        if args.rank != 0:
            while not graph_dict_file.exists():
                time.sleep(1)
            with open(str(graph_dict_file), "rb") as f:
                graph_dict = pickle.load(f)
                logger.info('Load graph dict: {}'.format(graph_dict_file)) if logger is not None else None
            self.graphs = graph_dict['graphs']
            self.shortest_paths = graph_dict['shortest_paths']
            # self.shortest_distances = graph_dict['shortest_distances']
            del graph_dict
        else:
            if graph_dict_file.exists():
                with open(str(graph_dict_file),"rb") as f:
                    graph_dict = pickle.load(f)
                    logger.info('Load graph dict: {}'.format(graph_dict_file)) if logger is not None else None
                self.graphs = graph_dict['graphs']
                self.shortest_paths = graph_dict['shortest_paths']
                # self.shortest_distances = graph_dict['shortest_distances']
                del graph_dict
            else:
                self.graphs = load_nav_graphs(connectivity_dir, self.scans)
                self.shortest_paths = {}
                for scan, G in self.graphs.items():  # compute all shortest paths
                    self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
                self.shortest_distances = {}
                for scan, G in self.graphs.items():  # compute all shortest paths
                    self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
                graph_dict = {'graphs': self.graphs, 'shortest_paths': self.shortest_paths,
                              'shortest_distances': self.shortest_distances}
                graph_dict_file.parent.mkdir(parents=True,exist_ok=True)
                with open(str(graph_dict_file), "wb") as f:
                    pickle.dump(graph_dict, f)
                logger.info('Save graph dict to: {}'.format(graph_dict_file)) if logger is not None else None

        #### Image Dir ####
        self.img_dir = Path(self.config.IMG_DIR).resolve()
        self.image_preprocess = image_transform(
            self.config.vision_encoder.image_size,
            is_train=False,
            mean=None,
            std=None
        )

        random.seed(self.seed)
        # random.shuffle(self.data) if self.training else None

        if logger is not None:
            logger.info('[INFO] %s loaded with %d instructions, using splits: %s' % (
                self.__class__.__name__, len(self.data), self.split))

    def load_instr_datasets(self,anno_file):
        """
         :return: example of self.data[0]
            'scan': 'VLzqgDo317F'
            'instr_id': '6250_0'
            'path_id': 6250
            'path': []
            'heading':
            'instruction':
            'heading':
            'distance':
        """
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
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        scan = item['scan']
        viewpoint_ids = item['path']
        heading = item['heading']
        env = EnvBatch(
            shortest_paths=self.shortest_paths[scan],
            navigable_loc=self.navigable_loc[scan],
            img_dir=self.img_dir,
            batch_size=1
        )
        env.newEpisodes([scan],[viewpoint_ids[0]],[heading])
        return {
            'sample_idx': index,
            'path_id': item['path_id'],
            'instr_id': item['instr_id'],
            'scan': scan,
            'paths': viewpoint_ids,
            'heading': heading,
            'env': env,
            'instruction': item['instruction']
        }

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


def build_dataloader(dataset,batch_size,dist=False,training=True,workers=0,seed=None):
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    shuffle = False # (sampler is None) and training

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=shuffle, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    dataloader.num_batches = len(dataloader)
    return dataset, dataloader, sampler