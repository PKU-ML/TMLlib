import os

from utils.dataset import prepare_dataloader
from utils.misc import set_all_seed, get_logger
from utils.args import get_args

from trainers import ReBATTrainer
from params import ReBATParam


def main():

    args = get_args(ReBATParam)

    param = ReBATParam(args)

    set_all_seed(param.seed)

    os.makedirs(param.save_dir, exist_ok=True)

    logger = get_logger(param.log_file)

    logger.info(args)

    train_dataloader, val_dataloader = prepare_dataloader(param)

    trainer = ReBATTrainer(param, train_dataloader, val_dataloader, logger)

    trainer.run()


if __name__ == "__main__":
    main()
