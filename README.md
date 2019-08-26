# Model Slicing

![version](https://img.shields.io/badge/version-v2.0-brightgreen)
![python](https://img.shields.io/badge/python-3.6.5-blue)
![pytorch](https://img.shields.io/badge/pytorch-0.4.1-blue)

This repository contains code for the paper [Model Slicing for Supporting Complex Analytics with Elastic Inference Cost and Resource Constraints](https://arxiv.org/abs/1904.01831).
Model Slicing for runtime accuracy-efficiency trade-offs with ease: slice a sub-layer that is composed of preceding groups of the full layer controlled by the slice rate r during each forward pass. Only the activated parameters and groups of the current layer are required in memory and participate in computation.

<img src="https://user-images.githubusercontent.com/14588544/62041132-908ddf00-b22d-11e9-8275-0167fcd27b74.png" width=25%/>

We illustrate a dense layer with slice rate r=0.5 (activated groups highlighted in blue) and r=0.75 (additional involved groups highlighted in green).

### The repo includes:

1. example models (/models)
2. codes for model slicing training (train.py)
3. codes to support model slicing (models/model_slicing.py)
    * upgrading neural networks to support elastic inference simply by calling one function (models/model_slicing/upgrade_dynamic_layers)

### Training
1. Dependencies
    * python 3.6.5
    * pytorch 0.4.1 (may require minor revision for later versions)
    * torchvision 0.2.1 (may require minor revision for later versions)
2. Model Training

```
Example training code:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 256 --lr 0.1  --dataset imagenet --data_dir /data/ --log_freq 50

Please check help info in argparse.ArgumentParser (train.py) for more details 
```

3. Upgrading Model

```
model = upgrade_dynamic_layers(model, args.groups, args.sr_list)

    * groups:   group_num in Group Normalization, e.g. 8/16, default 8
    * sr_list:  slice rate list, e.g. [1.0, 0.75, 0.5, 0.25]
```

### Contact
To ask questions or report issues, please open an issue here or can directly send [us](mailto:shaofeng@comp.nus.edu.sg) an email.
