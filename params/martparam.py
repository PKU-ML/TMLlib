from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class MARTParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)
        parser.add_argument("--mart_beta", type=float)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.mart_beta: float = float(args.mart_beta)
