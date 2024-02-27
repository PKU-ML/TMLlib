from argparse import Namespace, ArgumentParser
from pathlib import Path
import torch
from datetime import datetime


class TrainParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--model", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--save_freq", type=int)
        parser.add_argument("--taskname", type=str)
        parser.add_argument("--resume", type=bool)
        parser.add_argument("--device", type=str)

    def __init__(self, args: Namespace) -> None:
        self.epochs: int = int(args.epochs)
        self.model: str = str(args.model)
        self.seed: int = int(args.seed)
        self.save_freq: int = int(args.save_freq)
        if hasattr(args, "taskname") and args.taskname is not None and len(args.taskname) > 0:
            self.taskname: Path = Path(args.taskname)
        else:
            self.taskname: Path = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.resume: bool = bool(args.resume)
        self.device: str = str(args.device)

        self.save_dir = Path("save_file") / self.taskname
        self.log_file = self.save_dir / 'train.log'
        self.device: torch.device = torch.device(args.device)
