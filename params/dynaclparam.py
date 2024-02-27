from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class DynACLParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--swap_param", type=float)
        parser.add_argument("--twoLayerProj", type=bool)
        parser.add_argument("--val_frequency", type=int)
        parser.add_argument("--reload_frequency", type=int)

    def __init__(self, args: Namespace) -> None:
        BaseATParam.__init__(self, args)
        self.optimizer = str(args.optimizer)
        self.swap_param = float(args.swap_param)
        self.twoLayerProj = bool(args.twoLayerProj)
        self.val_frequency = int(args.val_frequency)
        self.reload_frequency = int(args.reload_frequency)
