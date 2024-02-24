from argparse import Namespace, ArgumentParser
from pathlib import Path


class LRScheduleParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--lr_schedule", type=str)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--lr_max", type=float)
        parser.add_argument("--lr_one_drop", type=float)
        parser.add_argument("--lr_drop_epoch", type=int)

    def __init__(self, args: Namespace) -> None:
        self.lr_schedule: str = str(args.lr_schedule)
        self.epochs: int = int(args.epochs)
        self.lr_max: float = float(args.lr_max)
        self.lr_one_drop: float = float(args.lr_one_drop) if self.lr_schedule == 'onedrop' else 0.0
        self.lr_drop_epoch: int = int(args.lr_drop_epoch) if self.lr_schedule == 'onedrop' else 0
