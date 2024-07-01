# TMLlib
A Trustworthy Machine Learning Algorithm Library. This repo collects the official implementations of works done by [Yisen Wang](https://yisenwang.github.io/) and his students.

# TODO 
- [x] Our repo is released! 
- [] Add README.
- [] Add package dependency and scripts to start the code.
- [] Report benchmark results for reference.

# Environment
The environment in which we build this repo is given in ```requirements.txt```. Run the below command to configure the environment.
```sh
  pip install -r requirements.txt
```
# List of contents
- [Valina Adversarial Training](https://arxiv.org/abs/1706.06083)
- [MART](https://openreview.net/forum?id=rklOg6EFwS)
- [AWP](https://arxiv.org/abs/2004.05884)
- [Generalist](https://arxiv.org/abs/2303.13813)
- [DynACL](https://arxiv.org/abs/2303.01289)
- [ReBAT](https://arxiv.org/abs/2310.19360)

# Implemented Methods & Usage
### 1. [Valina Adversarial Training](https://arxiv.org/abs/1706.06083)
To start the code
```sh
  python main_baseat.py --yaml config/baseat.yaml
```
Parameters are given in ```config/baseat.yaml```.


### 2. Improving Adversarial Training Requires Revisiting Mis-classified Examples ([MART](https://openreview.net/forum?id=rklOg6EFwS))
To start the code
```sh
  python main_mart.py --yaml config/mart.yaml
```
Parameters are given in ```config/mart.yaml```.


### 3. Adversarial Weight Perturbations Help Robust Generalization ([AWP](https://arxiv.org/abs/2004.05884))
To start the code
```sh
  python main_awp.py --yaml config/awp.yaml
```
Parameters are given in ```config/awp.yaml```.


### 4. Generalist: Decoupling natural and robust generalization ([Generalist](https://arxiv.org/abs/2303.13813))
To start the code
```sh
  python main_generalist.py --yaml config/generalist.yaml
```
Parameters are given in ```config/generalist.yaml```.


### 5. Rethinking the effect of data augmentation in adversarial contrastive learning ([DynACL](https://arxiv.org/abs/2303.01289))
To start the code
```sh
  python main_dynacl.py --yaml config/dynacl.yaml // Pretraining
  python main_dynacllinear.py --yaml config/dynacllinear.yaml // Linear-Probing
```
Parameters are given in ```config/dynacl.yaml```.

### 6. Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective ([ReBAT](https://arxiv.org/abs/2310.19360))
To start the code
```sh
  python main_rebat.py --yaml config/rebat.yaml 
```
Parameters are given in ```config/rebat.yaml```.

# Evaluate
To evaluate the accuracy/robustness of pre-trained model
```
  python main_eval.py --yaml config/eval.yaml 
```
Parameters are given in ```config/eval.yaml```.

# Acknowledgement
We refer some of the links below while building this repo. We sincerely acknowlegde their great work!
- https://github.com/Harry24k/adversarial-attacks-pytorch
- https://github.com/fra31/auto-attack
- https://github.com/MadryLab/robustness
- https://github.com/pytorch/vision/blob/main/torchvision/models
