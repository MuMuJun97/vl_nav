export PYTHONPATH=/mnt/petrelfs/zhaolin/vln/mp3d/Matterport3DSimulator-Centos7/build:$PYTHONPATH
export http_proxy=http://zhaolin:EavQCi89*7@10.1.8.50:33128/
export https_proxy=http://zhaolin:EavQCi89*7@10.1.8.50:33128/
# --cand_loss \
# --hist_loss \
NUM_GPU=2
torchrun --nnodes=1 --nproc_per_node=$NUM_GPU r2r/main_nav.py \
--dataset r2r \
--cand_loss \
--hist_loss \
--output_dir ../../../vl_nav_debug \
--world_size $NUM_GPU \
--seed 0 \
--tokenizer bert \
--enc_full_graph --graph_sprels \
--fusion local \
--num_l_layers 9 \
--num_x_layers 4 \
--num_pano_layers 2 \
--max_action_len 15 \
--max_instr_len 200 \
--batch_size 2 \
--lr 5e-5 \
--iters 20000 \
--log_every 1000 \
--optim adamW \
--expert_policy spl \
--train_alg dagger \
--ml_weight 0.2 \
--feat_dropout 0.4 \
--dropout 0.5 \
--gamma 0. \
--features evaclip_4096 \
--image_feat_size 4096 \
--angle_feat_size 4 \
--root_dir ../../build/duet
