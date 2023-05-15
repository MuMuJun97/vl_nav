torchrun --nnodes=1 --nproc_per_node=1 r2r_train.py \
--tokenizer_path /mnt/lustre/share_data/huangshijia/alpaca \
--run_name baseline \
--cfg_file tools/cfgs/datasets/s2_r2r_dataset.yaml \
--batch_size 1 \
--learning_rate 5e-5 \
--vision_encoder_path "ViT-B-16" \
--warmup_steps 500 \
--workers 4 \
--logging_steps 200