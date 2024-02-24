from argparse import Namespace, ArgumentParser
from pathlib import Path

from .attackparam import AttackParam
from .dataparam import DataParam
from .lrscheduleparam import LRScheduleParam
from .optimparam import OptimParam
from .trainparam import TrainParam


class AWPParam(DataParam, LRScheduleParam, OptimParam, TrainParam, AttackParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        DataParam.add_argument(parser)
        LRScheduleParam.add_argument(parser)
        OptimParam.add_argument(parser)
        TrainParam.add_argument(parser)
        AttackParam.add_argument(parser)
        parser.add_argument("--l2", type=int)
        parser.add_argument("--l1", type=int)
        parser.add_argument("--mixup", type=bool)
        parser.add_argument("--mixup_alpha", type=float)
        parser.add_argument("--awp_gamma", type=float)
        parser.add_argument("--awp_warmup", type=int)

    def __init__(self, args: Namespace) -> None:
        super(DataParam,       self).__init__(args)
        super(LRScheduleParam, self).__init__(args)
        super(OptimParam,      self).__init__(args)
        super(TrainParam,      self).__init__(args)
        super(AttackParam,     self).__init__(args)
        self.l2: int = int(args.l2)
        self.l1: int = int(args.l1)
        self.mixup: bool = bool(args.mixup)
        self.mixup_alpha: float = float(args.mixup_alpha)
        self.awp_gamma: float = float(args.awp_gamma)
        self.awp_warmup: int = int(args.awp_warmup if self.awp_gamma > 0.0 else int(args.epochs) + 1000000)
