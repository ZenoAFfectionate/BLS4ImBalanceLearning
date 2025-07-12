from .cifar10 import CIFAR10_LT
from .cifar100 import CIFAR100_LT
from .imagenet import ImageNet_LT

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def balance_dataloader(dataset_name, batch_size, augment=True):

    if dataset_name == "MNIST":
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dataset = torchvision.datasets.MNIST(root='../datasets', train=True, transform=transform_train, download=True)
        test_dataset = torchvision.datasets.MNIST(root='../datasets', train=False, transform=transform_test)
        n_classes = 10

    elif dataset_name == "FashionMNIST":
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dataset = torchvision.datasets.FashionMNIST(root='../datasets', train=True, transform=transform_train, download=True)
        test_dataset  = torchvision.datasets.FashionMNIST(root='../datasets', train=False, transform=transform_test)
        n_classes = 10

    elif dataset_name == "CIFAR10":
        if augment == True:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True,  transform=transform_train, download=True)
        test_dataset  = torchvision.datasets.CIFAR10(root='../datasets', train=False, transform=transform_test)
        n_classes = 10

    elif dataset_name == "CIFAR100":
        if augment == True:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='../datasets', train=True,  transform=transform_train, download=True)
        test_dataset  = torchvision.datasets.CIFAR100(root='../datasets', train=False, transform=transform_test)
        n_classes = 100

    elif dataset_name == "ImageNet-1K":
        if augment == True:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        test_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = torchvision.datasets.ImageFolder(root='/opt/datasets/ImageNet/train', transform=train_transform)
        test_dataset  = torchvision.datasets.ImageFolder(root='/opt/datasets/ImageNet/val', transform=test_transform)
        n_classes = 1000

    elif dataset_name == "ImageNet-21K":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = torchvision.datasets.ImageFolder(root='/opt/datasets/ImageNet-21K/train', transform=train_transform)
        test_dataset  = torchvision.datasets.ImageFolder(root='/opt/datasets/ImageNet-21K/val', transform=test_transform)
        n_classes = 21000

    else: raise ValueError("No such dataset: {dataset_name}")

    # print(f'The size of train_dataset is {len(train_dataset)}   The size of test_dataset  is {len(test_dataset)}')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, n_classes


def imbalance_dataloader(dataset_name, batch_size, imbalance_factor):

    config = Config()
    # config.sampler.dual_sample = True
    # config.sampler.dual_sample.enable = True
    # config.sampler.dual_sample.type = "long-tailed"
    # config.sampler.type = "weighted sampler"
    # config.sampler.weighted_sampler.type = "reverse"

    if dataset_name == 'CIFAR10':
        dataset = CIFAR10_LT(config, distributed=False, imb_factor=imbalance_factor, batch_size=batch_size)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        dataset = CIFAR100_LT(config, distributed=False, imb_factor=imbalance_factor, batch_size=batch_size)
        num_classes = 100
    elif dataset_name == 'ImageNet-1K':
        dataset = ImageNet_LT(config, distributed=False, batch_size=batch_size)
        num_classes = 1000
    elif dataset_name == 'INa2018':
        dataset = INa2018()
        num_classes = None
    else: raise ValueError("No such dataset: {dataset_name}")

    train_loader = dataset.train_instance
    valid_loader = dataset.eval
    cls_num_list = dataset.cls_num_list
    return train_loader, valid_loader, cls_num_list, num_classes


class Config:
    """ Simple configuration class for dataset """
    def __init__(self, **kwargs):
        self.sampler = type('', (), {})()
        self.sampler.type = None
        self.sampler.dual_sample = type('', (), {})()
        self.sampler.dual_sample.enable = False
        self.sampler.dual_sample.type = None
        self.sampler.weighted_sampler = type('', (), {})()
        self.sampler.weighted_sampler.type = None
        
        # 更新配置
        for key, value in kwargs.items():
            if '.' in key:
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    if not hasattr(obj, part):
                        setattr(obj, part, type('', (), {})())
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)