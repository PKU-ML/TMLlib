from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class GeneralistParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
