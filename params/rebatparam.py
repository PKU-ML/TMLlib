from argparse import Namespace, ArgumentParser
from pathlib import Path

from .attackparam import AttackParam
from .dataparam import DataParam
from .lrscheduleparam import LRScheduleParam
from .optimparam import OptimParam
from .trainparam import TrainParam


class ReBATParam(DataParam, LRScheduleParam, OptimParam, TrainParam, AttackParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        DataParam.add_argument(parser)
        LRScheduleParam.add_argument(parser)
        OptimParam.add_argument(parser)
        TrainParam.add_argument(parser)
        AttackParam.add_argument(parser)

    def __init__(self, args: Namespace) -> None:
        super(DataParam,       self).__init__(args)
        super(LRScheduleParam, self).__init__(args)
        super(OptimParam,      self).__init__(args)
        super(TrainParam,      self).__init__(args)
        super(AttackParam,     self).__init__(args)
