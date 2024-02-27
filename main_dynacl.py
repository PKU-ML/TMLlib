import os

from utils.dataset import prepare_dataloader
from utils.ssldataset import prepare_ssldataloader
from utils.misc import set_all_seed, get_logger
from utils.args import get_args

from trainers import DynACLTrainer
from params import DynACLParam


def main():

    args = get_args(DynACLParam)

    param = DynACLParam(args)

    set_all_seed(param.seed)

    os.makedirs(param.save_dir, exist_ok=True)

    logger = get_logger(param.log_file)

    logger.info(args)

    train_dataloader, _, test_dataloader = prepare_ssldataloader(param)

    trainer = DynACLTrainer(param, train_dataloader, test_dataloader, logger)

    trainer.run()


if __name__ == "__main__":
    main()
