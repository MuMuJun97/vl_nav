export PYTHONPATH=/mnt/petrelfs/zhaolin/vln/mp3d/Matterport3DSimulator-Centos7/build:$PYTHONPATH
export http_proxy=http://zhaolin:EavQCi89*7@10.1.8.50:33128/
export https_proxy=http://zhaolin:EavQCi89*7@10.1.8.50:33128/

torchrun --nnodes=1 --nproc_per_node=1 --master_port 22008 duet/llm_vl/training_merge.py \
    --run_name _opt_object_demo2_re --cfg_file tools/cfgs/exp/multi.yaml --batch_size 8 \
    --num_epochs 20 --tokenizer_path facebook/opt-iml-1.3b --lr 3e-5 --precision fp32 \
    --use_eva_clip True --image_feat_size 4096 --img_ft_file eva_clip_4096