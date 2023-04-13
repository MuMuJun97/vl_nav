# Matterport3D Navigation Dataset

## Training script
```shell
# S2
## 1. srun --partition=OpenDialogLab_S2 --gres=gpu:8 --ntasks-per-node=1 -n1 -c 64 --mem-per-cpu=40G --pty bash
## 2. srun --partition=OpenDialogLab_S2 --gres=gpu:1 --ntasks-per-node=1 -n1 -c 32 --mem-per-cpu=40G --pty bash
## 3. torchrun --nnodes=1 --nproc_per_node=8 train_net.py --tokenizer_path /mnt/lustre/zhaolin/vln/llm/models --cfg_file tools/cfgs/datasets/s2_imgdatasets.yaml --batch_size 2 --vision_encoder_path "ViT-B-16" --cross_attn_every_n_layers 8 --warmup_steps 1200 --num_epochs 20 --workers=4 --logging_steps 100
## 4. torchrun --nnodes=1 --nproc_per_node=8 train_net.py --tokenizer_path /mnt/lustre/zhaolin/vln/llm/models --cfg_file tools/cfgs/datasets/s2_imgdatasets.yaml --batch_size 2 --vision_encoder_path "ViT-B-16" --cross_attn_every_n_layers 8 --warmup_steps 1200 --num_epochs 20 --workers=4 --logging_steps 1000
## 5. generate:
  torchrun --nnodes=1 --nproc_per_node=1 --master_port 33534 eval_net.py --text_generate --split val_seen --tokenizer_path /mnt/lustre/zhaolin/vln/llm/models --cfg_file tools/cfgs/datasets/s2_imgdatasets.yaml --batch_size 2 --vision_encoder_path "ViT-L-14" --cross_attn_every_n_layers 8 --run_name Train1 --warmup_steps 1200 --num_epochs 40 --workers=0 --logging_steps 1000 --learning_rate 3e-5 --weight_decay 0.05  --text_generate --split val_seen --generate_start_index 6000 --generate_nums 200
torchrun --nnodes=1 --nproc_per_node=8 train_net.py \ 
  --tokenizer_path /mnt/lustre/zhaolin/vln/llm/models \ # LLaMa-7B
  --cfg_file tools/cfgs/datasets/s2_imgdatasets.yaml \
  --batch_size 2 \
  --vision_encoder_path "ViT-B-16" \
  --cross_attn_every_n_layers 8 \
  --warmup_steps 1200 \
  --num_epochs 20 \
  --workers=4 \
  --logging_steps 100
  
torchrun --nnodes=1 --nproc_per_node=2 train_net.py \
  --tokenizer_path /mnt/lustre/zhaolin/vln/llm/models \
  --cfg_file tools/cfgs/datasets/s2_imgdatasets.yaml \
  --batch_size 2

## 4. TEST
  - time cost: (single GPU, batch size 1)
    --vision_encoder_path "ViT-B-16"
      CLIP-Encoder: 12 image views, ~14.80 ms
      Model(CLIP+LLaMa-7B) Forward: ~61.54 ms
    --vision_encoder_path "ViT-L-14"
      CLIP-Encoder: 12 image views, ~45.13 ms
      Model(CLIP+LLaMa-7B) Forward: ~88.55 ms

## 5. --cross_attn_every_n_layers 4/8
__Comment: How often to apply cross attention after transformer layer. 
  raw LLaMa-7B: 32 LlamaDecoderLayers
(1) --cross_attn_every_n_layers 4: 32//4=8 GatedCrossAttentionBlocks
(2) --cross_attn_every_n_layers 8: 32//8=4 GatedCrossAttentionBlocks
https://github.com/mlfoundations/open_flamingo/issues/129#issuecomment-1492884192 
```

----

![Node1](./tests/imgs/1.png)
![Node2](./tests/imgs/2.png)
![Node3](./tests/imgs/3.png)
![Node4](./tests/imgs/4.png)

# 1. Dataset
```python
# dataset/base_dataset.py
train_dataset = BaseDataset(
    config=dataset_cfg.Dataset, # tools/cfgs/datasets/datasets.yaml
    split=args.split # 'train'
)
```

## 2. Data Examples
```shell
# for SOON Dataset
['<image>Question: What are the attributes of the target object? Answer: This is a brand new white, rectangular wooden table.<|endofchunk|></s>',
 '<image>Question: What is the relationship between the target object and other objects in the room? Answer: It is above a few chairs, under a pot of flowers.<|endofchunk|></s>']

# for Fine-grained Dataset
['<image>Question: What is the next step I should take based on the instruction: go through the bedroom? Answer: You should go in direction 1.<|endofchunk|></s>',
 '<image>Question: What is the next step I should take based on the instruction: and stop in front of the spa bath? Answer: You should go in direction 3.<|endofchunk|></s>']

```

## TODO Prompt
```python
# dataset/preprocess_data.py
promptQAs = {
    'soon_qa': [
        "What are the attributes of the target object?",                                     # 0
        "What is the relationship between the target object and other objects in the room?", # 1
        "Which room or area is the target object in?",                                       # 2
        "What is the relationship between the target room and other neighboring rooms?",     # 3
        "What is the navigation instruction for the current scene?",                         # 4 full instruction
        "What is the target object of navigation?",                                          # 5 target object
    ],
    'image2text': [
        "What direction should I turn after reaching <Image>?",
        "What should I do after reaching <Image>?",
        "What is the next step after reaching <Image>?",
    ],
    'image+text': [
        "Based on the image <Image> and the instruction <Instruction>, what is the next step I should take?",
        "Using the image <Image> and the instruction <Instruction>, what is the next landmark I should look for?",
        "Given the instruction <Instruction>, which direction should I turn based on the image <Image>?",
        "What is the next direction to follow after reaching the location shown in <Image>, given the instruction <Instruction>?",
        "From the current image, which direction should I turn based on the instruction <Instruction>?",
        "Which direction should I turn based on the instruction <Instruction>?", # 5
        "What is the next step I should take based on the instruction: <Instruction>?", # 6
    ],
    'image+viewpoint': [
        "From the current image <Image>, what should I do to reach <Direction-{}>?",
        "After reaching the current image <Image>, what should I do to get to <Direction-{}>?",
        "What is the next step after reaching <Image> and facing <Direction-{}>?",
    ],
    # TODO image-text format, from https://github.com/mlfoundations/open_flamingo
    "open_flamingo": [
        "<image>{text}<|endofchunk|>{tokenizer_eos_token}",

        # <image_0>..<image_11>{text}<|endofchunk|>{tokenizer_eos_token}
        "".join(["<image_{}>".format(i) for i in range(12)]) + "{text}<|endofchunk|>{tokenizer_eos_token}"
    ],
    "answer+viewpoint": [
        "You should walk towards <Direction-{ViewID}>.",
        "You should head towards <Direction-{ViewID}>.",
        "You should go in direction {ViewID}.", # 2
        "You should {STOP}.", # 3
    ],
}

```
