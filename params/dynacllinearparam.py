from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class DynACLLinearParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)
        parser.add_argument("--optimizer", type=str)
        parser.add_argument("--swap_param", type=float)
        parser.add_argument("--twoLayerProj", type=bool)

    def __init__(self, args: Namespace) -> None:
        BaseATParam.__init__(self, args)
        self.optimizer = str(args.optimizer)
        self.swap_param = float(args.swap_param)
        self.twoLayerProj = bool(args.twoLayerProj)
