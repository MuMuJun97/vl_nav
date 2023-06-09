import pickle
import random
from collections import defaultdict
import numpy as np
from .preprocess_data import (
    preprocess_soon_v1,preprocess_fr2r,
    promptQAs,
    generate_direction_from_mp3d
)
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

class BaseDataset(torch_data.Dataset):
    def __init__(self,config, split, training=True, logger=None, in_memory=True, **kwargs):
        super().__init__()
        self.config = config
        self.split = split
        self.training = training
        self.logger = logger
        self.data_dir = Path(config.DATA_DIR).resolve()
        root_dir = Path(__file__).parent.parent.resolve()

        #################### Source Dataset ####################
        # read Matterport3D navigableLocations
        self.navigable_loc = self.get_navigable_Locations()

        # generate_direction_from_mp3d(self.navigable_loc)

        self.source_data = config.SOURCE
        print('[INFO] ********* use dataset : {} *********'.format(self.source_data))
        self.data = []
        if 'SOON' in self.source_data:
            soon_file = root_dir / config.SOON.DIR / config.SOON.SPLIT[split]
            # read SOON data
            self.soon_data = preprocess_soon_v1(
                soon_file,
                self.navigable_loc,
            )
            self.data += self.soon_data
        else:
            self.soon_data = []
        if 'FR2R' in self.source_data:
            fr2r_file = root_dir / config.FR2R.DIR / config.FR2R.SPLIT[split]
            # read Fine-grained data
            self.fr2r_data,self.fr2r_answers_counter = preprocess_fr2r(
                fr2r_file,
                self.navigable_loc,
            )
            self.data += self.fr2r_data
        else:
            self.fr2r_data = []
            self.fr2r_answers_counter = None

        # '</s>'
        self.tokenizer_eos_token = self.config.tokenizer.eos_token
        self.prompt = self.config.tokenizer.prompt

        if self.config.get('IMG_DIR',None) is not None:
            self.in_memory = in_memory
            self.img_ft_file = None
            self.obj_ft_file = None
            self.img_dir = Path(self.config.IMG_DIR).resolve()
            from open_clip.transform import image_transform
            self.image_preprocess = image_transform(
                self.config.vision_encoder.image_size,
                is_train=False,
                mean=None,
                std=None
            )
        else:
            # read image features
            self.in_memory = in_memory
            if self.in_memory:
                self._feature_store = {}
            self.img_ft_file = self.data_dir / self.config.Img_Features_File_Map

            if config.With_Object_Feats:
                self.obj_ft_file = self.data_dir / self.config.Object_Features_File_Map
            else:
                self.obj_ft_file = None
            self.img_dir = None

        if kwargs.get('generate_start_index',None) is not None:
            self.generate_start_index = kwargs['generate_start_index']
        else:
            self.generate_start_index = None

        if kwargs.get('save_img',None) is not None:
            self.save_img = kwargs['save_img']
        else:
            self.save_img = None

        print('Dataset Initialize')

    def __len__(self):
        return len(self.soon_data) + len(self.fr2r_data)

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

    def read_image(self, scan, viewpoint, index):
        img_file = self.img_dir / scan / '{}_{}.png'.format(scan,viewpoint)
        assert img_file.exists()
        img = cv2.imread(str(img_file)) # BRG

        if self.save_img is not None:
            save_dir = self.img_dir.parent / 'samples'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_file = save_dir / "{}_{}_{}.png".format(index,scan,viewpoint)
            cv2.imwrite(str(save_file),img)

        imgs = np.hsplit(img,12)

        images = [
            self.image_preprocess(
                Image.fromarray(
                    s[:,:,::-1] # BRG2RGB
                )
            ).unsqueeze(0)
            for s in imgs
        ]
        images = torch.cat(images, dim=0) # [12,3,224,224]

        if self.training:
            # apply random horizontal flip and color jitter
            images = torchvision.transforms.RandomHorizontalFlip(p=0.5)(images)
            images = torchvision.transforms.ColorJitter(brightness=0.5, hue=0.3)(images)

        return images, None, None

    def get_image_data(self, scan, viewpoint, index):
        if self.img_dir is not None:
            return self.read_image(scan, viewpoint, index)
        else:
            return self.get_scan_viewpoint_feature(scan, viewpoint)

    def get_scan_viewpoint_feature(self, scan, viewpoint, enable_HFOV=False, only_img_feats=True):
        """
        Args:
            scan: matterport 3d scene
            viewpoint: the prefix/name of current node/viewpoint
        Returns:
            view_fts: [num_views, img_feat_dim] [12,1768]
                num_views=12 direction, img_feat_dim=image_feat_size(0:768) # +image_prob_size(768:1768)
            # obj_fts: [num_objects, obj_feat_dim] [num_objects,2048+1601=3649]
            #     obj_feat_dim=obj_feat_size(0:2048)+obj_prob_size(2048:)
            # obj_attrs:
            #     'bboxes': [num_objects,4]
            #     'directions': [num_objects,2]
            #     'obj_ids': [num_objects,]
            #     'sizes': [num_objects,2]
        """
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(str(self.img_ft_file), 'r') as f:
                view_fts = f[key][...].astype(np.float32)
                view_fts = view_fts[:36]

            # select 12 view image
            if not enable_HFOV:
                view_fts = view_fts[12:24]
            if only_img_feats:
                view_fts = view_fts[:,:self.config.image_feat_size]

            obj_attrs = {}
            obj_ft_lens = self.config.obj_feat_size+self.config.obj_prob_size
            obj_fts = np.zeros((0, obj_ft_lens), dtype=np.float32)

            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.config.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.config.max_objects]
                        obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32)
                        obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                        obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                        obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)
        return view_fts, obj_fts, obj_attrs

    def generate_input_text(self,question,answer):
        task_desc_type = self.prompt.task_type
        task_description = promptQAs['task_description'][task_desc_type]
        prompt = self.prompt.template[self.prompt.type]

        # environment = "".join(["<image>" if i==1 else "<image>" for i in view_mask])
        use_image_id = self.prompt.image_id
        environment = "".join(["{}. <image> ".format(i) if use_image_id else "<image>" for i in range(12)])
        environment = environment + "."
        question = " ".join(question.split())
        answer = " ".join(answer.split())

        if self.prompt.type == "common":
            input_text = prompt.format(
                task_description=task_description,
                environment=environment,
                question=question,
                answer=answer,
                tokenizer_eos_token=self.tokenizer_eos_token
            )
        elif self.prompt.type == "flamingo":
            input_text = prompt.format(
                task_description=task_description,
                environment=environment,
                question=question,
                answer=answer,
                endofchunk='<|endofchunk|>',
                tokenizer_eos_token=self.tokenizer_eos_token
            )
        else:
            raise NotImplementedError

        # whitespace cleanup
        input_text = (
            input_text.replace(" <|endofchunk|>", "<|endofchunk|>")
            .replace("<image> ", "<image>")
            .replace(" <image>", "<image>")
            .replace(" Question", ".Question")
            .replace(" Answer", "?Answer")
        )

        return input_text

    def get_data_dict(self, item, scan, index):
        data_dict = dict()

        ### [1] Fine-grained R2R Dataset
        if index >= len(self.soon_data):

            question = item['qa']['question']
            answer = item['qa']['answer']

            # for fine-grained dataset: sub-path <-> sub-instruction
            viewpoint = item['viewpoint']
            view_fts, obj_img_fts, obj_attrs = self.get_image_data(scan, viewpoint, index)
            # ViewpointNext = item['ViewpointNext']  # next direction {0..11}, -1 means STOP

            if item['qa_type'] == 'instr2view':
                instr2view_id = answer.replace('Answer:','')
                data_dict['instr2view_id'] = instr2view_id
            else:
                data_dict['instr2view_id'] = ''

        ### [2] SOON Dataset
        else:
            data_dict['instr2view_id'] = ''
            question = item['qa']['question']
            answer = item['qa']['answer']
            vp_index = -1  # end viewpoint

            # for soon dataset: end viewpoint is the target location <-> instructions
            viewpoint = item['path'][vp_index] # item['path'][-1] is the goal location
            view_fts, obj_img_fts, obj_attrs = self.get_image_data(scan, viewpoint, index)

        input_text = self.generate_input_text(
            question=question,
            answer=answer,
        )

        if self.img_dir is not None:
            data_dict.update({
                'input_text': input_text,
                'imgs': view_fts,
            })
        else:
            data_dict.update({
                'input_text': input_text,
                'img_feats': view_fts[:, :self.config.image_feat_size],
                'obj_feats': obj_img_fts[:, :self.config.obj_feat_size] if self.obj_ft_file is not None else None,
            })
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
                if key in ['input_text','sample_idx','instr2view_id']:
                    ret[key] = val
                elif key in ['imgs']:
                    ret[key] = torch.stack(val, 0)
                else:
                    np_val = np.stack(val, axis=0) # ['img_feats'] (B,12,768)
                    ret[key] = torch.from_numpy(np_val)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        # ret['batch_size'] = batch_size
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

    # if dist:
    #     shuffle = (sampler is None) and training
    # else:
    #     shuffle = False
    shuffle = (sampler is None) and training

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=shuffle, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )
    dataloader.num_batches = len(dataloader)
    return dataset, dataloader, sampler