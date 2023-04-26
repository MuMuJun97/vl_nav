import numpy as np
import math
import json
from pathlib import Path
import random
import os
import networkx as nx
import cv2
import copy
import pickle

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
    def newEpisode(self, scanId, viewpointId, heading, elevation):
        self.state = SimState(scanId=scanId,viewpointId=viewpointId,heading=heading,elevation=elevation)
        nav_dict = self.navigable_loc[scanId][viewpointId]
        state_viewIndex = self.state.viewIndex - self.headingCount
        for k,v in nav_dict.items():
            if v['pointId'] == state_viewIndex:
                self.state.set_navigableLocations(v)

    def getState(self):
        return self.state


class EnvBatch(object):
    def __init__(self,navigable_loc,connectivity_dir,img_dir,batch_size):
        self.img_dir = img_dir
        # self.sims = []
        # for i in range(batch_size):
        #     sim = new_simulator(connectivity_dir=connectivity_dir)
        #     self.sims.append(sim)

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
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        img_dir = Path(self.config.IMG_DIR).resolve()

        self.env = EnvBatch(
            navigable_loc=self.navigable_loc,
            connectivity_dir=connectivity_dir,
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