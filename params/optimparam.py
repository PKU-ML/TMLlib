from argparse import Namespace, ArgumentParser
from pathlib import Path


class OptimParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--lr_max", type=float)
        parser.add_argument("--weight_decay", type=float)
        parser.add_argument("--momentum", type=float)
        parser.add_argument("--l1", type=float)
        parser.add_argument("--l2", type=float)

    def __init__(self, args: Namespace) -> None:
        self.lr_max: float = float(args.lr_max)
        self.weight_decay: float = float(args.weight_decay)
        self.momentum: float = float(args.momentum)
        self.l1: float = float(args.l1)
        self.l2: float = float(args.l2)
