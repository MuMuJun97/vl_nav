import numpy as np
import math
from easydict import EasyDict
import yaml
import json
from dataset.utils.visualize_mp3d import show_12_images, mp3d_view_r2r
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
import sys
from collections import defaultdict
from dataset.process_multi_data import (
    load_r2r_data,load_reverie_data,
    load_soon_data,load_fr2r_data
)


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


def setHeading(heading, heading_count=12, elevation=0):
    M_PI = math.pi
    elevation_increment = M_PI / 6.0 # 30°
    state_heading = heading % (M_PI*2.0)
    while state_heading < 0.0:
        state_heading += math.pi*2.0
    # Snap heading to nearest discrete value
    headingIncrement = M_PI * 2.0 / heading_count
    heading_step = round(state_heading / headingIncrement)
    if heading_step == heading_count:
        heading_step = 0
    state_heading = heading_step * headingIncrement
    # Snap elevation to nearest discrete value (disregarding elevation limits)
    if elevation < -elevation_increment/2.0:
        elevation = -elevation_increment
        viewIndex = heading_step
    elif elevation > elevation_increment/2.0:
        elevation = elevation_increment
        viewIndex = heading_step + 2*heading_count
    else:
        elevation = 0.0
        viewIndex = heading_step + heading_count
    return viewIndex,state_heading,elevation

class MP3DDataset(torch_data.Dataset):
    def __init__(self, config, split='train', training=True, logger=None, batch_size=2, seed=0):
        self.config = config
        self.split = split
        self.logger = logger
        self.batch_size = batch_size
        self.training = training
        self.seed = seed
        self.img_dir = Path(config.IMG_DIR).resolve()

        _root_dir = Path(__file__).parent.parent.resolve()
        self.data = dict()
        self.alldata = []
        self.all_index = dict()
        for source in config.SOURCE:
            if source == 'R2R':
                _anno_file = _root_dir/config.R2R.DIR/config.R2R.SPLIT[split]
                self.data['r2r'] = load_r2r_data(anno_file=_anno_file)
            elif source == 'REVERIE':
                _anno_file = _root_dir/config.REVERIE.DIR/config.REVERIE.SPLIT[split]
                self.data['reverie'] = load_reverie_data(anno_file=_anno_file)
            elif source == 'SOON':
                _anno_file = _root_dir/config.SOON.DIR/config.SOON.SPLIT[split]
                self.data['soon'] = load_soon_data(anno_file=_anno_file)
            elif source == "FR2R":
                _anno_file = _root_dir/config.FR2R.DIR/config.FR2R.SPLIT[split]
                self.data['fr2r'] = load_fr2r_data(anno_file=_anno_file)
            else:
                NotImplementedError
        start_index = 0
        end_index = 0
        if 'r2r' in self.data.keys():
            self.alldata += self.data['r2r']
            end_index += len(self.data['r2r'])
            self.all_index.update({i:'r2r' for i in range(end_index)})
            start_index += len(self.data['r2r'])
        if 'reverie' in self.data.keys():
            self.alldata += self.data['reverie']
            end_index += len(self.data['reverie'])
            self.all_index.update({i:'reverie' for i in range(start_index,end_index)})
            start_index += len(self.data['reverie'])
        if 'soon' in self.data.keys():
            self.alldata += self.data['soon']
            end_index += len(self.data['soon'])
            self.all_index.update({i:'soon' for i in range(start_index,end_index)})
            start_index += len(self.data['soon'])
        if 'fr2r' in self.data.keys():
            self.alldata += self.data['fr2r']
            end_index += len(self.data['fr2r'])
            self.all_index.update({i:'fr2r' for i in range(start_index,end_index)})

        # mp3d navigable dict
        self.navigable_loc = self.get_navigable_Locations()
        connectivity_dir = str(_root_dir / 'data/connectivity')
        graph_dict_file = _root_dir / 'build/R2R/R2R_{}_graph_dict.pkl'.format(self.split)
        if graph_dict_file.exists():
            with open(str(graph_dict_file), "rb") as f:
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
            graph_dict_file.parent.mkdir(parents=True, exist_ok=True)
            with open(str(graph_dict_file), "wb") as f:
                pickle.dump(graph_dict, f)
            logger.info('Save graph dict to: {}'.format(graph_dict_file)) if logger is not None else None

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

    def read_image(self, scan, viewpoint, split=True):
        img_file = self.img_dir / scan / '{}_{}.png'.format(scan,viewpoint)
        assert img_file.exists()
        panoramic_img = cv2.imread(str(img_file)) # BRG
        if split:
            panoramic_img = np.hsplit(panoramic_img, 12)
        return panoramic_img

    def __len__(self):
        return len(self.alldata)

    def __getitem__(self, index):
        if self.all_index[index] == 'r2r':
            item = self.alldata[index]

            # mp3d_view_r2r(item,self.navigable_loc)

            print(item['instruction'])
            view_index, _, _ = setHeading(
                heading=item['heading']
            )
            cur_view_index = view_index - 12 # 当前view所在view_idx,
            images = []
            imgs = self.read_image(scan=item['scan'],viewpoint=item['path'][0])
            vrgb = show_12_images(imgs, view_id=cur_view_index)
            cv2.imshow('all', vrgb)
            # images.append(vrgb)
            # cv2.imshow('all',vrgb)
            cv2.waitKey(0)

            for vi,vp in enumerate(item['path']):
                if vp == item['path'][-1]:
                    raise NotImplementedError
                else:
                    next_vp = item['path'][vi+1]
                    view_idx = self.navigable_loc[item['scan']][vp][next_vp]['pointId']
                    heading = (view_idx % 12) * math.radians(30)
                    view_index, _, _ = setHeading(
                        heading=heading
                    )
                    # cur_view_index = view_index - 12
                    imgs = self.read_image(scan=item['scan'],viewpoint=vp)

                    # cv2.imshow('view', images[-1])
                    vrgb = show_12_images(imgs,view_id=view_idx)

                    images.append(vrgb)

                    cv2.imshow('all', vrgb)
                    traj_img = np.concatenate(images,axis=0)
                    cv2.imshow('traj',traj_img)
                    cv2.waitKey(0)



        print('One')



def test():
    _cfg_file = Path(__file__).parent.parent.resolve() / "tools/cfgs/datasets/mp3d_multi_data.yaml"
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(_cfg_file).resolve()))))
    dataset = MP3DDataset(
        config=global_cfg.Dataset,
        split='train',
        training=True,
        logger=None,
        batch_size=2,
        seed=0
    )
    for item in dataset:
        item

if __name__ == '__main__':
    test()