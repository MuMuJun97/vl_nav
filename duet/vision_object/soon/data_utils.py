import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import copy

import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from utils.data import angle_feature


def normalize_angle(x):
    '''convert radians into (-pi, pi]'''
    pi2 = 2 * math.pi
    x = x % pi2 # [0, 2pi]
    if x > math.pi:
        x = x - pi2
    return x

def convert_heading(x):
    return x % (2 * math.pi) / (2 * math.pi)   # [0, 2pi] -> [0, 1)

def convert_elevation(x):
    return (normalize_angle(x) + math.pi) / (2 * math.pi)   # [0, 2pi] -> [0, 1)

def load_instr_datasets(anno_dir, dataset, splits):
    assert dataset == 'soon'

    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            new_data = []
            # load instructions
            # '/home/zlin/vln/llm/vl_nav/build/duet/SOON/annotations/obj_bert_enc/train_enc_pseudo_obj_ade30k_label.jsonl'
            input_file = os.path.join(anno_dir, 'obj_bert_enc', '%s_enc_pseudo_obj_ade30k_label.jsonl'%split)
            if not os.path.exists(input_file):
                input_file = os.path.join(anno_dir, 'obj_bert_enc', '%s_enc.jsonl'%split)
            with jsonlines.open(input_file, 'r') as f:
                for item in f:
                    item['end_image_ids'] = [x['image_id'] for x in item['bboxes']]
                    item['image_id_to_obj_label'] = {x['image_id']: x.get('pseudo_label', None) for x in item['bboxes']}
                    new_bboxes = {}
                    for bbox in item['bboxes']:
                        new_bboxes[bbox['image_id']] = bbox
                    item['bboxes'] = new_bboxes
                    new_data.append(item)
        else:   # augmented data (TODO)
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        # Join
        data += new_data
    return data[:20]

def construct_instrs(anno_dir, dataset, splits, instr_type='full', tokenizer=None, max_instr_len=512):
    assert dataset == 'soon'
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = copy.deepcopy(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr[instr_type]
            new_item['instr_encoding'] = item['instr_encodings'][j][instr_type][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data


class ObjectFeatureDB(object):
    def __init__(self, obj_ft_file, obj_feat_size):
        self.obj_feat_size = obj_feat_size
        self.obj_ft_file = obj_ft_file
        self._feature_store = {}
        self.env = lmdb.open(obj_ft_file, readonly=True)

    def __del__(self):
        self.env.close()

    def load_feature(self, scan, viewpoint, max_objects=None):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            obj_fts, obj_attrs = self._feature_store[key]
        else:
            with self.env.begin() as txn:
                obj_data = txn.get(key.encode('ascii'))
            if obj_data is not None:
                obj_data = msgpack.unpackb(obj_data)
                obj_fts = obj_data['fts'][:, :self.obj_feat_size].astype(np.float32) 
                obj_attrs = {
                    'directions': obj_data['2d_centers'],
                    'obj_ids': obj_data['obj_ids'],
                    'bboxes': np.array(obj_data['xyxy_bboxes']), 
                }
            else:
                obj_attrs = {}
                obj_fts = np.zeros((0, self.obj_feat_size), dtype=np.float32)
            self._feature_store[key] = (obj_fts, obj_attrs)

        if max_objects is not None:
            obj_fts = obj_fts[:max_objects]
            obj_attrs = {k: v[:max_objects] for k, v in obj_attrs.items()}
        return obj_fts, obj_attrs

    def get_object_feature(
        self, scan, viewpoint, base_heading, base_elevation, angle_feat_size,
        max_objects=None
    ):
        obj_fts, obj_attrs = self.load_feature(scan, viewpoint, max_objects=max_objects)
        obj_ang_fts = np.zeros((len(obj_fts), angle_feat_size), dtype=np.float32)
        obj_loc_fts = np.zeros((len(obj_fts), 3), dtype=np.float32)
        obj_directions, obj_ids = [], []
        if len(obj_fts) > 0:
            for k, obj_ang in enumerate(obj_attrs['directions']):
                obj_ang_fts[k] = angle_feature(
                    obj_ang[0] - base_heading, obj_ang[1] - base_elevation, angle_feat_size
                )
                x1, y1, x2, y2 = obj_attrs['bboxes'][k]
                h = y2 - y1
                w = x2 - x1
                obj_loc_fts[k, :2] = [h/224, w/224]
                obj_loc_fts[k, 2] = obj_loc_fts[k, 0] * obj_loc_fts[k, 1]
            obj_directions = [[convert_heading(x[0]), convert_elevation(x[1])] for x in obj_attrs['directions']]
            obj_ids = obj_attrs['obj_ids']
        return obj_fts, obj_ang_fts, obj_loc_fts, obj_directions, obj_ids