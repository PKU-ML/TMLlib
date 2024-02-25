from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List


class LRScheduleParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--lr_schedule", type=str)
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--lr_max", type=float)
        parser.add_argument("--epoch_list", type=str)
        parser.add_argument("--lr_list", type=str)

    def __init__(self, args: Namespace) -> None:
        self.lr_schedule: str = str(args.lr_schedule)
        self.epochs: int = int(args.epochs)
        self.lr_max: float = float(args.lr_max)

        if isinstance(args.epoch_list, str):
            self.epoch_list = list(eval(args.epoch_list))
        else:
            self.epoch_list = list(args.epoch_list)
        self.epoch_list: List[int] = [int(e) for e in self.epoch_list]

        if isinstance(args.lr_list, str):
            self.lr_list = list(eval(args.lr_list))
        else:
            self.lr_list = list(args.lr_list)
        self.lr_list: List[float] = [float(e) for e in self.lr_list]
