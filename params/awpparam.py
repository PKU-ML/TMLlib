from argparse import Namespace, ArgumentParser
from pathlib import Path

import numpy as np
import math


class AWPParam():
    def __init__(self, args: Namespace) -> None:
        self.taskname = args.taskname
        self.model = args.model
        self.l2 = args.l2
        self.l1 = args.l1
        self.batch_size = args.batch_size
        # self.batch_size_test = args.batch_size_test
        self.data_dir = args.data_dir
        self.epochs = args.epochs
        self.lr_schedule = args.lr_schedule
        self.lr_max = args.lr_max
        self.lr_one_drop = args.lr_one_drop
        self.lr_drop_epoch = args.lr_drop_epoch
        self.attack = args.attack
        self.epsilon = args.epsilon
        self.attack_iters = args.attack_iters
        # self.attack_iters_test = args.attack_iters_test
        self.restarts = args.restarts
        self.pgd_alpha = args.pgd_alpha
        self.fgsm_alpha = args.fgsm_alpha
        self.norm = args.norm
        self.fgsm_init = args.fgsm_init
        self.seed = args.seed
        self.half = args.half
        self.width_factor = args.width_factor
        self.resume = args.resume
        self.cutout = args.cutout
        self.cutout_len = args.cutout_len
        self.mixup = args.mixup
        self.mixup_alpha = args.mixup_alpha
        self.eval = args.eval
        self.chkpt_iters = args.chkpt_iters
        self.awp_gamma = args.awp_gamma
        self.awp_warmup = args.awp_warmup

        if self.awp_gamma <= 0.0:
            self.awp_warmup = np.infty

        self.epsilon = (self.epsilon / 255.)
        self.pgd_alpha = (self.pgd_alpha / 255.)

        if self.attack == 'free':
            self.epochs = int(math.ceil(self.epochs / self.attack_iters))

        self.save_dir = Path("save_file") / self.taskname
        self.log_file = self.save_dir / ('eval.log' if self.eval else 'train.log')
