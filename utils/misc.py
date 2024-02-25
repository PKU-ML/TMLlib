import errno
import os
import numpy as np
import random


import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init as init
import logging


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Save checkpoint


def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def torch_accuracy(output, target, topk=(1,)):
    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    if len(target.size()) == 1:
        is_correct = pred.eq(target.view(1, -1).expand_as(pred))
    elif len(target.size()) == 2:
        is_correct = pred.eq(target.max(1)[1].expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


def set_all_seed(seed_num: int):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


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


def get_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
    return logger
