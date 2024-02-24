from argparse import Namespace, ArgumentParser
from pathlib import Path


class TrainParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--model", type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--save_freq", type=int)
        parser.add_argument("--taskname", type=str)
        parser.add_argument("--resume", type=bool)
        parser.add_argument("--eval", type=bool)

    def __init__(self, args: Namespace) -> None:
        self.epochs: int = int(args.epochs)
        self.model: str = str(args.model)
        self.seed: int = int(args.seed)
        self.save_freq: int = int(args.save_freq)
        self.taskname: Path = Path(args.taskname)
        self.resume: bool = bool(args.resume)
        self.eval: bool = bool(args.eval)
