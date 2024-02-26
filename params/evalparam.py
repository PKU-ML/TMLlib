from argparse import Namespace, ArgumentParser
from pathlib import Path
import torch

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
        parser.add_argument("--device", type=str)

    def __init__(self, args: Namespace) -> None:
        DataParam.__init__(self, args)
        AttackParam.__init__(self, args)
        self.model: str = str(args.model)
        self.seed: int = int(args.seed)
        self.taskname: Path = Path(args.taskname)
        self.ckptname: Path = Path(args.ckptname)
        self.device: str = str(args.device)

        self.save_dir = Path("save_file") / self.taskname
        self.log_file = self.save_dir / 'eval.log'
        self.ckpt_file = self.save_dir / self.ckptname
        self.device: torch.device = torch.device(args.device)
