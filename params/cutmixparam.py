from argparse import Namespace, ArgumentParser
from pathlib import Path


class CutmixParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--cutmix", type=bool)
        parser.add_argument("--cutmix_alpha", type=float)
        parser.add_argument("--cutmix_beta", type=float)

    def __init__(self, args: Namespace) -> None:
        self.cutmix: bool = bool(args.cutmix)
        self.cutmix_alpha: float = float(args.cutmix_alpha)
        self.cutmix_beta: float = float(args.cutmix_beta)
