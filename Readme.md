# Matterport3D Navigation Dataset

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
