import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
from easydict import EasyDict
from dataset.base_dataset import BaseDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--cfg_file', type=str, default="tools/cfgs/datasets/datasets.yaml", help='dataset configs')
    parser.add_argument('--img_feats', type=str, default="vit_imagenet", help='dataset configs')
    parser.add_argument('--obj_feats', type=str, default="butd_SOON", help='dataset configs')
    parser.add_argument('--split', type=str, default="train", help='dataset configs')

    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))
    dataset_cfg.Dataset.Img_Features_File_Map = dataset_cfg.Dataset.Img_Features_File_Map[args.img_feats]
    dataset_cfg.Dataset.Object_Features_File_Map = dataset_cfg.Dataset.Object_Features_File_Map[args.obj_feats]

    dataset = BaseDataset(config=dataset_cfg.Dataset,split=args.split)

    pbar = tqdm(dataset,desc="iterate dataset: ")
    for i, data_dict in enumerate(pbar):
        pass