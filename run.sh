torchrun --nnodes=1 --nproc_per_node=8 r2r_train.py \
--tokenizer_path /mnt/lustre/huangshijia.p/LLAMA_7B \
--run_name baseline7 \
--cfg_file tools/cfgs/datasets/s2_r2r_dataset.yaml \
--batch_size 1 \
--learning_rate 5e-5 \
--vision_encoder_path "ViT-B-16" \
--warmup_steps 500 \
--workers 4 \
--num_epochs 4 \
--save_ckpt_step 1 \
--logging_steps 200

torchrun --nnodes=1 --nproc_per_node=8 r2r_test.py \
--tokenizer_path /mnt/lustre/huangshijia.p/LLAMA_7B \
--run_name baseline7 \
--cfg_file tools/cfgs/datasets/s2_r2r_dataset.yaml \
--batch_size 1 \
--learning_rate 1e-5 \
--vision_encoder_path "ViT-B-16" \
--warmup_steps 500 \
--workers 4 \
--logging_steps 200 \
--num_epochs 1 \
--split val_seen \
--resume_from_checkpoint /mnt/petrelfs/huangshijia.p/MM/vl_nav_output/baseline8/checkpoint_3.pt


torchrun --nnodes=1 --nproc_per_node=8 r2r_test.py \
--tokenizer_path /mnt/lustre/huangshijia.p/LLAMA_7B \
--run_name baseline1 \
--cfg_file tools/cfgs/datasets/s2_r2r_dataset.yaml \
--batch_size 1 \
--learning_rate 5e-5 \
--vision_encoder_path "ViT-B-16" \
--warmup_steps 500 \
--workers 4 \
--logging_steps 200 \
--num_epochs 1 \
--split val_seen \
--resume_from_checkpoint /mnt/lustre/huangshijia.p/MM/vl_nav_output/baseline1/checkpoint_2.pt