import os

from utils.log import get_logger
from utils.dataset import prepare_dataloader
from utils.misc import set_all_seed
from utils.args import get_args

from trainers import AWPTrainer
from params import AWPParam


def main():

    args = get_args(AWPParam)

    param = AWPParam(args)

    set_all_seed(param.seed)

    os.makedirs(param.save_dir, exist_ok=True)

    logger = get_logger(param.log_file)

    logger.info(param)

    train_dataloader, val_dataloader = prepare_dataloader(param)

    trainer = AWPTrainer(param, train_dataloader, val_dataloader, logger)

    trainer.run()


if __name__ == "__main__":
    main()
