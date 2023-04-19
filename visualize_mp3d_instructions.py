import json
import re
import cv2
import pickle
import yaml
from pathlib import Path
from easydict import EasyDict
import argparse
import random
import numpy as np
import torch
from dataset.preprocess_data import generate_direction_from_mp3d
from dataset.utils.visualize_mp3d import add_id_on_img,add_token_on_img


def read_args():
    output_dir = Path(__file__).parent.parent.parent.resolve() / "vl_nav_output"

    parser = argparse.ArgumentParser()
    parser.add_argument('--instr_path',
                        type=str,
                        default="build/data/mp3d_12view_instructions_checkpoint_10.json",
                        help='Matterport3D: generated instructions file path'
                        )
    parser.add_argument('--save_dir', type=str, default='/media/zlin/2CD830B2D8307C60/Dataset/features/generated')
    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/datasets/imgdatasets.yaml", help='dataset configs')
    parser.add_argument('--split', type=str, default="train", help='train, val, test')
    parser.add_argument('--output_dir', type=str, default=str(output_dir), help='output dir')
    parser.add_argument('--img_feats', type=str, default="vit_imagenet", help='dataset configs')
    parser.add_argument('--obj_feats', type=str, default="butd_SOON", help='object features')

    args = parser.parse_args()

    ############# CONFIGURATION #############
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    global_cfg.Dataset.Img_Features_File_Map = global_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    global_cfg.Dataset.Object_Features_File_Map = global_cfg.Dataset.Object_Features_File_Map[args.obj_feats]
    args.enable_imgdataset = False if global_cfg.Dataset.get('IMG_DIR', None) is None else True
    args.max_length = global_cfg.Dataset.tokenizer.max_length

    return args, global_cfg


def get_navigable_Locations(config):
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
    mp3d_nav_file = Path(__file__).parent.resolve() / config.MP3D_NAV
    with open(str(mp3d_nav_file), "rb") as f:
        res_dict = pickle.load(f)
    return res_dict


class MP3DViewDataset(object):
    def __init__(self, instr_data, mp3d_data, img_dir, save_dir=None):
        super().__init__()
        self.data = instr_data
        self.mp3d_dict = mp3d_data
        self.img_dir = Path(img_dir).resolve()
        if save_dir is not None:
            self.save_dir = Path(save_dir).resolve()
        else:
            self.save_dir = None

    def __len__(self):
        return len(self.data)

    def read_image(self, scan, viewpoint):
        img_file = self.img_dir / scan / '{}_{}.png'.format(scan, viewpoint)
        assert img_file.exists()
        img = cv2.imread(str(img_file))  # BRG
        images = np.hsplit(img, 12)
        return images

    def __getitem__(self, index):
        item = self.data[index]
        scan = self.mp3d_dict[index]['scan']
        viewpoint = self.mp3d_dict[index]['viewpoint']
        sample_idx = self.mp3d_dict[index]['sample_idx']
        assert sample_idx == int(item[0])

        images = self.read_image(scan,viewpoint)

        question = item[1]['_gt_question_text']
        instruction = item[1]['generate_answer_text'].\
            replace('<|endofchunk|>','').\
            replace('</s>','').\
            replace('<s>','').\
            replace('<unk>','')
        view_id = list(map(int, re.findall('\d+', question)))[0]
        assert 0 <= view_id <= 11
        print(view_id)

        vrgb = []
        for i in range(len(images)):
            st_idx = 60
            ed_idx = images[0].shape[1] - st_idx
            curimg = images[i][:, st_idx:ed_idx, :]
            id_img = add_id_on_img(curimg, str(i))
            id_img[:, -1, :] = 0
            if i == view_id:
                id_img = add_token_on_img(id_img,token=str(i),color=(0,0,255),height=int(id_img.shape[0]/3*2))
            vrgb.append(id_img)
        vrgb = np.concatenate(vrgb, axis=1).astype(np.uint8)
        vrgb = add_token_on_img(vrgb,token=instruction)
        cv2.imshow('RGB', vrgb)
        cv2.waitKey(0)

        if self.save_dir is not None:
            save_path = self.save_dir / '{}_{}_{}.png'.format(scan,viewpoint,view_id)
            cv2.imwrite(str(save_path),vrgb)

def main(args, config):
    assert Path(args.instr_path).resolve().exists()
    with open(str(args.instr_path), "r") as f:
        instr_data = json.load(f)
    instr_data = sorted(instr_data.items(), key=lambda item: int(item[0]))

    navigable_loc = get_navigable_Locations(config.Dataset)
    mp3d_data = generate_direction_from_mp3d(navigable_loc)
    assert len(mp3d_data) == len(instr_data)

    dataset = MP3DViewDataset(
        instr_data=instr_data,
        mp3d_data=mp3d_data,
        img_dir=config.Dataset.IMG_DIR,
        save_dir=args.save_dir,
    )

    for data in dataset:
        pass


if __name__ == '__main__':
    opts, global_cfg = read_args()
    main(opts, global_cfg)
