from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from params import DataParam
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

    # TODO: More dataset

    train_dataset: Dataset = DATASET_CLASS(root=param.data_dir, train=True, download=True, transform=transform_train)

    test_dataset: Dataset = DATASET_CLASS(root=param.data_dir, train=False, download=True, transform=transform_test)

    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True, num_workers=param.num_workers)

    test_dataloader: DataLoader = DataLoader(test_dataset, batch_size=param.batch_size, shuffle=False, num_workers=param.num_workers)

    return train_dataloader, test_dataloader


'''

def get_data_batches(data_dir: str, batch_size: int, do_cutout: bool = False, cutout_len: int = 0):
    transforms = [Crop(32, 32), FlipLR()]
    if do_cutout:
        transforms.append(Cutout(cutout_len, cutout_len))
    # if do_val:
    #     try:
    #         dataset = torch.load("cifar10_validation_split.pth")
    #     except:
    #         print("Couldn't find a dataset with a validation split, did you run "
    #               "generate_validation.py?")
    #         return
    #     val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
    #     val_batches = Batches(val_set, train_batch_size, shuffle=False, num_workers=2)
    # else:
    dataset = cifar(data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
                         dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=2)
    
    return train_batches, test_batches


def cifar(root, num_classes=10):

    # TODO deprecated this
    if num_classes == 10:
        train_set = CIFAR10(root=root, train=True, download=True)
        test_set = CIFAR10(root=root, train=False, download=True)
    elif num_classes == 100:
        train_set = CIFAR100(root=root, train=True, download=True)
        test_set = CIFAR100(root=root, train=False, download=True)
    else:
        raise NotImplementedError("Only CIFAR-10 and CIFAR-100 are available!")
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }


#####################
# data loading
#####################

class Batches:
    # TODO deprecated this
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle,
            drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x, y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)
'''
