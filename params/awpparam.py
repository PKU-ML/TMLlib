from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class AWPParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)
        parser.add_argument("--awp_gamma", type=float)
        parser.add_argument("--awp_warmup", type=int)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.awp_gamma: float = float(args.awp_gamma)
        self.awp_warmup: int = int(args.awp_warmup if self.awp_gamma > 0.0 else int(args.epochs) + 1000000)
