import os

from utils.dataset import prepare_dataloader
from utils.misc import set_all_seed, get_logger
from utils.args import get_args

from trainers import ModelEvaler
from params import EvalParam


def main():

    args = get_args(EvalParam)

    param = EvalParam(args)

    set_all_seed(param.seed)

    os.makedirs(param.save_dir, exist_ok=True)

    logger = get_logger(param.log_file)

    logger.info(args)

    _, val_dataloader = prepare_dataloader(param)

    trainer = ModelEvaler(param, val_dataloader, logger)

    trainer.run()


if __name__ == "__main__":
    main()
