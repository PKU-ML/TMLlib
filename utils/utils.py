import numpy as np
import torch
import random


def set_all_seed(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)


def moving_average(net1, net2, decay_rate=0., update_bn=True):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= decay_rate
        param1.data += param2.data * (1 - decay_rate)

    if update_bn:
        for module1, module2 in zip(net1.modules(), net2.modules()):
            if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
                module1.running_mean *= decay_rate
                module1.running_mean += module2.running_mean * (1 - decay_rate)
                module1.running_var *= decay_rate
                module1.running_var += module2.running_var * (1 - decay_rate)
                module1.num_batches_tracked = module2.num_batches_tracked
