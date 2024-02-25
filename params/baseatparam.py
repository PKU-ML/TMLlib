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
        super(DataParam,       self).__init__(args)
        super(TrainParam,      self).__init__(args)
        super(OptimParam,      self).__init__(args)
        super(LRScheduleParam, self).__init__(args)
        super(AttackParam,     self).__init__(args)
        super(CutmixParam,     self).__init__(args)
