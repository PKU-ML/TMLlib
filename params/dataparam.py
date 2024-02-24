from argparse import Namespace, ArgumentParser
from pathlib import Path


class DataParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--dataset", type=str)
        parser.add_argument("--data_dir", type=str)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--num_workers", type=int)

    def __init__(self, args: Namespace) -> None:
        self.dataset: str = str(args.dataset)
        self.data_dir: Path = Path(args.data_dir)
        self.batch_size: int = int(args.batch_size)
        self.num_workers: int = int(args.num_workers)
        # TODO more aug params?
        # TODO Type check?
