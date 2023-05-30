## Large Language and Vision Model for Embodied AI

### 1. Pretraining
- TODO
### 2. Fine-tuning
- TODO
### 3. Tasks in Matterport3D Simulator

**3.1 Matterport3DSimulator in Centos7**
```shell script
# install packages
pip install line-profiler==3.2.2

# python 3.8 
$ export PYTHONPATH=/mnt/petrelfs/zhaolin/vln/mp3d/Matterport3DSimulator-Centos7/build:$PYTHONPATH

# test
$ python -c "import MatterSim;print(MatterSim.__file__)"
>>> /mnt/petrelfs/zhaolin/vln/mp3d/Matterport3DSimulator-Centos7/build/MatterSim.cpython-38-x86_64-linux-gnu.so
```

**3.2 download data**

Download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0) to `build/duet`, including processed annotations, features and pretrained models of REVERIE, SOON, R2R and R4R datasets. Put the data in `build/duet` directory.
```shell script
mkdir -p build/duet
cd build/duet
wget https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0 -O datasets.zip
unzip datasets.zip
```

**3.3 Fine-tuning from VLN-DUET**
```shell script
# to fine-tune the model in R2R Dataset
cd duet/map_nav_src
python r2r/main_nav.py --dataset r2r --output_dir ../../../vl_nav_output --world_size 1 --seed 0 --tokenizer bert --enc_full_graph --graph_sprels --fusion dynamic --num_l_layers 9 --num_x_layers 4 --num_pano_layers 2 --max_action_len 15 --max_instr_len 200 --batch_size 8 --lr 1e-5 --iters 200000 --log_every 1000 --optim adamW --expert_policy spl --train_alg dagger --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5 --gamma 0. --features vitbase --image_feat_size 768 --angle_feat_size 4 --root_dir ../../build/duet
```

**3.4 Multi-GPU train DUET**
```shell script
# train on 8 GPU, batch_size_per_gpu=8, total_batch_size=64, lr=5e-5 (or 1e-5), epochs=20
torchrun --nnodes=1 --nproc_per_node=8 duet/map_nav_src/llm_training.py \
 --run_name {RUN_NAME} --cfg_file tools/cfgs/datasets/s2_r2r_dataset.yaml \
 --batch_size 8 --num_epochs 20 --lr 5e-5
# some results:
eval on val_unseen epoch 0: sr: 22.19, oracle_sr: 62.37
eval on val_unseen epoch 1: sr: 27.55, oracle_sr: 61.27
eval on val_unseen epoch 2: sr: 34.52, oracle_sr: 62.54
eval on val_unseen epoch 3: sr: 40.39, oracle_sr: 56.59
...
eval on val_unseen epoch 7: sr: 45.07, oracle_sr: 65.01
```

**3.5 Multi-GPU train with Alpaca-7B**
```shell script
torchrun --nnodes=1 --nproc_per_node=8 duet/map_nav_src_llm/llm_training.py \
 --run_name {RUN_NAME} --cfg_file tools/cfgs/datasets/s2_r2r_dataset.yaml \
 --batch_size 1 --num_epochs 20

```