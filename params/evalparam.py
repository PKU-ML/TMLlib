from argparse import Namespace, ArgumentParser
from pathlib import Path

from .dataparam import DataParam
from .attackparam import AttackParam


class EvalParam(DataParam, AttackParam):

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        DataParam.add_argument(parser)
        AttackParam.add_argument(parser)
        parser.add_argument("--model", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--taskname", type=str)
        parser.add_argument("--ckptname", type=str)

    def __init__(self, args: Namespace) -> None:
        super(DataParam,       self).__init__(args)
        super(AttackParam,     self).__init__(args)
        self.model: str = str(args.model)
        self.seed: int = int(args.seed)
        self.taskname: Path = Path(args.taskname)
        self.ckptname: Path = Path(args.ckptname)

        self.save_dir = Path("save_file") / self.taskname
        self.log_file = self.save_dir / 'eval.log'
        self.ckpt_file = self.save_dir / self.ckptname
