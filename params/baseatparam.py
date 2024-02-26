from argparse import Namespace, ArgumentParser
from pathlib import Path

from .dataparam import DataParam
from .trainparam import TrainParam
from .optimparam import OptimParam
from .lrscheduleparam import LRScheduleParam
from .attackparam import AttackParam
from .cutmixparam import CutmixParam


class BaseATParam(DataParam, TrainParam, OptimParam, LRScheduleParam, AttackParam, CutmixParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        DataParam.add_argument(parser)
        TrainParam.add_argument(parser)
        OptimParam.add_argument(parser)
        LRScheduleParam.add_argument(parser)
        AttackParam.add_argument(parser)
        CutmixParam.add_argument(parser)

    def __init__(self, args: Namespace) -> None:
        DataParam.__init__(self, args)
        TrainParam.__init__(self, args)
        OptimParam.__init__(self, args)
        LRScheduleParam.__init__(self, args)
        AttackParam.__init__(self, args)
        CutmixParam.__init__(self, args)
