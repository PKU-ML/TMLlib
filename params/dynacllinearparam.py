from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class DynACLLinearParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)
        parser.add_argument("--swap_param", type=float)
        parser.add_argument("--twoLayerProj", type=bool)
        parser.add_argument("--backbone_file", type=str)

    def __init__(self, args: Namespace) -> None:
        BaseATParam.__init__(self, args)
        self.swap_param = float(args.swap_param)
        self.twoLayerProj = bool(args.twoLayerProj)
        self.backbone_file = Path(args.backbone_file)
