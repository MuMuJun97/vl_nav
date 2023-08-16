# w/o gen loss
NUM_GPU=8
torchrun --nnodes=1 --nproc_per_node=$NUM_GPU r2r/main_nav.py \
--dataset r2r \
--output_dir ../../../vl_nav_output_llama_v4_2 \
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
--features vitbase \
--image_feat_size 768 \
--angle_feat_size 4 \
--root_dir ../../build/duet
