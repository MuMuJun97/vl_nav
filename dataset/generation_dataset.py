import pickle
import random
from collections import defaultdict
import numpy as np
from .preprocess_data import preprocess_soon_v1,preprocess_fr2r,promptQAs,generate_direction_from_mp3d
from tools.train import common_utils
from PIL import Image
import torch
import torchvision
import cv2
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
import h5py
import torch.utils.data as torch_data
from pathlib import Path
from .base_dataset import BaseDataset

class MP3DGenerationDataset(BaseDataset):
    def __init__(self, config, split, training=True, logger=None, in_memory=True, **kwargs):
        super().__init__(
            config=config,split=split,training=training,logger=logger,in_memory=in_memory,**kwargs
        )

        self.data = generate_direction_from_mp3d(self.navigable_loc)

    def __len__(self):
        return len(self.data)

    def get_data_dict(self, item, scan, index):
        viewpoint = item['viewpoint']
        view_fts, obj_img_fts, obj_attrs = self.get_image_data(scan, viewpoint, index)
        question = item['qa']['question_view2instr']
        input_text = self.generate_input_text(
            question=question,
            answer='Answer:',
        )
        if self.img_dir is not None:
            data_dict = {
                'input_text': input_text,
                'imgs': view_fts,
            }
        else:
            data_dict = {
                'input_text': input_text,
                'img_feats': view_fts[:, :self.config.image_feat_size],
                'obj_feats': obj_img_fts[:, :self.config.obj_feat_size] if self.obj_ft_file is not None else None,
            }
            if data_dict.get('obj_feats', None) is None:
                data_dict.pop('obj_feats')
        return data_dict

    def __getitem__(self, index):
        if self.generate_start_index is not None:
            index += self.generate_start_index

        item = self.data[index]
        scan = item['scan']

        data_dict = self.get_data_dict(item,scan,index)
        data_dict['sample_idx'] = index

        return data_dict


