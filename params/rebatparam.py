from argparse import Namespace, ArgumentParser
from pathlib import Path

from .baseatparam import BaseATParam


class ReBATParam(BaseATParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        BaseATParam.add_argument(parser)
        parser.add_argument('--decay_rate', default=0.999, type=float)
        parser.add_argument('--warmup_epochs', default=105, type=int)
        parser.add_argument('--stronger_attack', action='store_true', type=bool)
        parser.add_argument('--stronger_epsilon', default=10, type=float)
        parser.add_argument('--stronger_num_steps', default=12, type=int)
        parser.add_argument('--stronger_eval', action='store_true', type=bool)  # also use stronger attack during evaluation
        parser.add_argument('--use_reg_schedule', action='store_true', type=bool)  # if set to False, by default it stays constant as args.beta
        parser.add_argument('--boat_beta', type=float, default=1.0)
        parser.add_argument('--boat_beta_factor', type=float, default=1.5)  # multiply factor in piecewise schedule
        parser.add_argument('--reg_schedule', default='dependent', choices=['piecewise', 'dependent'], type=str)

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.decay_rate: float = float(args.decay_rate)
        self.warmup_epochs: int = int(args.warmup_epochs)
        self.stronger_attack: bool = bool(args.stronger_attack)
        self.stronger_epsilon: float = float(args.stronger_epsilon) / 255.0
        self.stronger_num_steps: int = int(args.stronger_num_steps)
        self.stronger_eval: bool = bool(args.stronger_eval)
        self.use_reg_schedule: bool = bool(args.use_reg_schedule)
        self.boat_beta: float = float(args.boat_beta)
        self.boat_beta_factor: float = float(args.boat_beta_factor)
        self.reg_schedule: str = str(args.reg_schedule)
