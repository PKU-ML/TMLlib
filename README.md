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

# Implemented Methods & Usage
### 1. Valina Adversarial Training (AT)
To start the code
```sh
  python main_baseat.py --yaml config/baseat.yaml
```
Parameters are given in ```config/baseat.yaml```.
Citation
```
  @article{madry2017towards,
  title={Towards deep learning models resistant to adversarial attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  journal={arXiv preprint arXiv:1706.06083},
  year={2017}}
```

### 2. Improving Adversarial Training Requires Revisiting Mis-classified Examples (MART)
To start the code
```sh
  python main_mart.py --yaml config/mart.yaml
```
Parameters are given in ```config/mart.yaml```.
Citation
```
  @inproceedings{wang2019improving,
  title={Improving adversarial robustness requires revisiting misclassified examples},
  author={Wang, Yisen and Zou, Difan and Yi, Jinfeng and Bailey, James and Ma, Xingjun and Gu, Quanquan},
  booktitle={International conference on learning representations},
  year={2019}
  }
```

### 3. Adversarial Weight Perturbations Help Robust Generalization (AWP)
To start the code
```sh
  python main_awp.py --yaml config/awp.yaml
```
Parameters are given in ```config/awp.yaml```.
Citation
```
  @article{wu2020adversarial,
  title={Adversarial weight perturbation helps robust generalization},
  author={Wu, Dongxian and Xia, Shu-Tao and Wang, Yisen},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={2958--2969},
  year={2020}
}
```

### 4. Generalist: Decoupling natural and robust generalization 
To start the code
```sh
  python main_generalist.py --yaml config/generalist.yaml
```
Parameters are given in ```config/generalist.yaml```.
Citation
```
@inproceedings{wang2023generalist,
  title={Generalist: Decoupling natural and robust generalization},
  author={Wang, Hongjun and Wang, Yisen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20554--20563},
  year={2023}
}
```

### 5. Rethinking the effect of data augmentation in adversarial contrastive learning (DynACL)
To start the code
```sh
  python main_dynacl.py --yaml config/dynacl.yaml // Pretraining
  python main_dynacllinear.py --yaml config/dynacllinear.yaml // Linear-Probing
```
Parameters are given in ```config/dynacl.yaml```.
Citation
```
@article{luo2023rethinking,
  title={Rethinking the effect of data augmentation in adversarial contrastive learning},
  author={Luo, Rundong and Wang, Yifei and Wang, Yisen},
  journal={arXiv preprint arXiv:2303.01289},
  year={2023}
}
```

# Evaluate
To evaluate the accuracy/robustness of pre-trained model
```
  python main_eval.py --yaml config/eval.yaml 
```
Parameters are given in ```config/eval.yaml```.
