from argparse import Namespace, ArgumentParser
from pathlib import Path


class MARTParam():
    def __init__(self, args: Namespace) -> None:
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.momentum = args.momentum
        self.no_cuda = args.no_cuda
        self.epsilon = args.epsilon
        self.num_steps = args.num_steps
        self.step_size = args.step_size
        self.beta = args.beta
        self.seed = args.seed
        self.log_interval = args.log_interval
        self.model = args.model
        self.save_freq = args.save_freq
