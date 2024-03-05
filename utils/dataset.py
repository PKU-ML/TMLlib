from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from params.dataparam import DataParam
from typing import Tuple

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])


def prepare_dataloader(param: DataParam) -> Tuple[DataLoader, DataLoader]:

    if param.dataset == "cifar10":
        DATASET_CLASS = CIFAR10
    elif param.dataset == "cifar100":
        DATASET_CLASS = CIFAR100
    else:
        raise ValueError("dataset not supported")

    # TODO: add more datasets

    train_dataset: Dataset = DATASET_CLASS(root=param.data_dir, train=True, download=True, transform=transform_train)

    test_dataset: Dataset = DATASET_CLASS(root=param.data_dir, train=False, download=True, transform=transform_test)

    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, num_workers=param.num_workers)

    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.num_workers)

    return train_dataloader, test_dataloader
