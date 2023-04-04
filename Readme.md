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
