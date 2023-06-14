import os
import math
from tqdm import tqdm
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import timm
import cv2
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from progressbar import ProgressBar
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp


# vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k
# timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k

# https://github.com/huggingface/pytorch-image-models/blob/fb4f220c2ef246e95c1221ce0de380f4b8b92957/README.md#dec-5-2022
# model	                                            top1	param_count	gmac	macts	hub
# vit_base_patch16_clip_384.laion2b_ft_in12k_in1k	87.2	86.9	    55.5	101.6	link
# timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k 输出feature=768
def build_feature_extractor(model_name='timm/vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=True).to(device)
    model = model.eval()
    # config = resolve_data_config({}, model=model)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return model, transforms, device


def process_features(proc_id, out_queue, scanvp_list, split):
    print('start proc_id: %d' % proc_id)
    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor()

    for idx in scanvp_list:
        images = []
        for j in range(5):
            fname = _anno_dir / "{}.{}.jpg".format('%04d' % idx, '%03d' % j)
            assert fname.exists()
            image = cv2.imread(str(fname))
            image = Image.fromarray(image[:, :, ::-1])
            images.append(image)
        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts, logits = [], []
        for k in range(0, len(images), 5):
            b_fts = model.forward_features(images[k: k + 5])
            # global_pool: Type of global pooling for final sequence (default: 'token').
            b_fts = b_fts[:, 0]
            b_fts = b_fts.data.cpu().numpy()
            fts.append(b_fts)

        fts = np.concatenate(fts, 0)

        out_queue.put((split, idx, fts))

    out_queue.put(None)


if __name__ == '__main__':

    anno_dir = Path("/home/zlin/vln/llm/habitat-lab/data/datasets/eqa/eqa_data/")
    eqa_nums = {
        # 'train': 11496,
        'val': 1950
    }
    num_workers = 6

    for split in eqa_nums.keys():
        scanvp_list = list(range(eqa_nums[split]))
        _anno_dir = anno_dir / split
        num_workers = min(num_workers, len(scanvp_list))
        num_data_per_worker = len(scanvp_list) // num_workers

        out_queue = mp.Queue()
        processes = []

        for proc_id in range(num_workers):
            sidx = proc_id * num_data_per_worker
            eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

            process = mp.Process(
                target=process_features,
                args=(proc_id, out_queue, scanvp_list[sidx: eidx], split)
            )
            process.start()
            processes.append(process)

        num_finished_workers = 0
        num_finished_vps = 0

        progress_bar = ProgressBar(maxval=len(scanvp_list))
        progress_bar.start()

        with h5py.File("/home/zlin/vln/llm/habitat-lab/data/datasets/eqa/eqa_vit_224_{}.hdf5".format(split), 'w') as outf:
            while num_finished_workers < num_workers:
                res = out_queue.get()
                if res is None:
                    num_finished_workers += 1
                else:
                    split, idx, fts = res
                    key = '%s_%s' % (split, idx)
                    outf.create_dataset(key, fts.shape, dtype='float', compression='gzip')
                    outf[key][...] = fts
                    num_finished_vps += 1
                    progress_bar.update(num_finished_vps)

        progress_bar.finish()
        for process in processes:
            process.join()