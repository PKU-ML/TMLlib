from argparse import Namespace, ArgumentParser
from pathlib import Path
import torch


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
        self.taskname: Path = Path(args.taskname)
        self.resume: bool = bool(args.resume)
        self.device: str = str(args.device)

        self.save_dir = Path("save_file") / self.taskname
        self.log_file = self.save_dir / 'train.log'
        self.device: torch.device = torch.device(args.device)
