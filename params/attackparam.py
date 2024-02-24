from argparse import Namespace, ArgumentParser
from pathlib import Path


class AttackParam():

    @staticmethod
    def add_argument(parser: ArgumentParser) -> None:
        parser.add_argument("--attack", type=str)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--epsilon", type=float)
        parser.add_argument("--step_size", type=float)
        parser.add_argument("--num_steps", type=int)
        parser.add_argument("--restart", type=int)

    def __init__(self, args: Namespace) -> None:
        self.attack: str = 'pgd' if str(args.attack) == 'pgd' else 'none'
        self.num_classes: int = int(args.num_classes)
        self.epsilon: float = float(args.epsilon) / 255.0 if self.attack == 'pgd' else 0.0
        self.step_size: float = float(args.step_size) / 255.0 if self.attack == 'pgd' else 0.0
        self.num_steps: int = int(args.num_steps) if self.attack == 'pgd' else 0
        self.restart: int = int(args.restart) if self.attack == 'pgd' else 0
