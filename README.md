# Model Slicing

![version](https://img.shields.io/badge/version-v2.2-brightgreen)
![python](https://img.shields.io/badge/python-3.8.3-blue)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-blue)
![singa](https://img.shields.io/badge/singa-3.1.0-orange)

This repository contains our PyTorch implementation of [Model Slicing for Supporting Complex Analytics with Elastic Inference Cost and Resource Constraints](https://arxiv.org/abs/1904.01831).
Model Slicing is a general *dynamic width technique* that enables neural networks to support 
budgeted inference, namely producing predictions within a prescribed computational budget by 
dynamically trading off accuracy for efficiency at runtime.

Budgeted inference is achieved by dividing each layer of the network into equal-sized *groups* 
of basic components (i.e., neurons in dense layers and channels in convolutional layers).
Technically, we use a single parameter called *slice rate **r*** to control the fraction of 
groups involved in computation for all layers at runtime, namely to control the width of 
the network in both training and inference.

In particular, the groups involved in computation always start from the first group, and 
contiguously to the dynamically determined last group indexed by the current slice rate.
E.g., a slice rate of 0.5 will select the first two groups in a layer of 4 groups 
as illustrated below.

<img src="https://user-images.githubusercontent.com/14588544/129475136-e5a2b85f-0dad-4a9c-865a-4920a0708769.png" width="200" />


### This repo includes:

1. representative models ([/models](https://github.com/nusdbsystem/model-slicing/blob/main/models))
2. codes for model slicing training ([train.py](https://github.com/nusdbsystem/model-slicing/blob/main/train.py))
3. codes for supporting *model slicing* functionalities ([models/model_slicing.py](https://github.com/nusdbsystem/model-slicing/blob/main/models/model_slicing.py))
    * upgrading a PyTorch model to support *model slicing* by calling one function ([models/model_slicing/upgrade_dynamic_layers](https://github.com/nusdbsystem/model-slicing/blob/main/models/model_slicing.py))

   
### Training
1. Dependencies
```
pip install -r requirements.txt
```

2. Model Training

```
Example training code:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --exp_name resnet_50 --net_type resnet --group 8 --depth 50 --sr_list 1.0 0.75 0.5 0.25 --sr_scheduler_type random_min_max --sr_rand_num 1 --epoch 100 --batch_size 256 --lr 0.1  --dataset imagenet --data_dir /data/ --log_freq 50

Please check help info in argparse.ArgumentParser (train.py) for configuration details.
```

3. One line to support *Model Slicing*

```
model = upgrade_dynamic_layers(model, args.groups, args.sr_list)

    * groups:   the number of groups for each layer, e.g. 8
    * sr_list:  slice rate list, e.g. [1.0, 0.75, 0.5, 0.25]
```

### Contact
To ask questions or report issues, you can directly drop us an [email](mailto:shaofeng@comp.nus.edu.sg).
