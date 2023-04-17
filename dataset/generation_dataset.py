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

        generate_direction_from_mp3d(self.navigable_loc)




