from .dataset import *


def get_transform(dataset):
    if dataset in ['cifar10', 'cifar100', 'imagenet32']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomRotation(22.5)
        ])

        test_transform = transforms.ToTensor()

    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_dataloader(dataset='cifar10', batch_size=128, num_workers=4, augment=True, data_path='data/'):
    DatasetClass = {'cifar10': datasets.CIFAR10,
                    'cifar100': datasets.CIFAR100,
                    'fashion': datasets.FashionMNIST,
                    'mnist': datasets.MNIST,
                    'imagenet32': IMAGENET32,
                    'graycifar10': GrayCIFAR10,
                    'emnist': datasets.EMNIST
                    }[dataset]

    n_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'fashion': 10,
        'mnist': 10,
        'imagenet32': 1000,
        'graycifar10': 10,
        'emnist': 10
    }[dataset]

    train_transform, test_transform = get_transform(dataset)

    if dataset in ['cifar10', 'cifar100', 'fashion', 'mnist', 'graycifar10']:
        trainset = DatasetClass(root=data_path, train=True, download=True,
                                transform=train_transform if augment else test_transform)

        testset = DatasetClass(root=data_path, train=False, download=True, transform=test_transform)

    elif dataset == 'emnist':
        trainset = DatasetClass(root=data_path, train=True, download=True,
                                transform=train_transform if augment else test_transform,
                                split='letters')
        testset = DatasetClass(root=data_path, train=False, download=True, transform=test_transform,
                               split='letters')

    elif dataset == 'imagenet32':
        trainset = DatasetClass(root=data_path, train=True,
                                transform=train_transform if augment else test_transform,
                                lbl_range=(0, 100), id_range=(1, 11))

        testset = DatasetClass(root=data_path, train=False, transform=test_transform,
                               lbl_range=(0, 100), id_range=(1, 11))

    else:
        raise NotImplementedError('Dataset is not implemented.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=augment, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, n_classes


def get_soft_label_dataloader(soft_label, dataset='cifar10', batch_size=128, num_workers=4, augment=True, data_path='data/'):
    DatasetClass = {'cifar10': SoftLabelCIFAR10,
                    'cifar100': SoftLabelCIFAR100,
                    'fashion': SoftLabelFashion,
                    'kmnist': SoftLabelKMNIST,
                    'mnist': SoftLabelMNIST,
                    'imagenet32': SoftIMAGENET32,
                    'graycifar10': SoftGrayCIFAR10,
                    'emnist': SoftLabelEMNIST
                    }[dataset]

    n_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'fashion': 10,
        'mnist': 10,
        'kmnist': 10,
        'imagenet32': 1000,
        'graycifar10': 10,
        'emnist': 10
    }[dataset]

    train_transform, test_transform = get_transform(dataset)

    if dataset in ['cifar10', 'cifar100', 'fashion', 'mnist', 'kmnist', 'graycifar10', 'emnist']:
        trainset = DatasetClass(root=data_path, train=True, download=True, soft_label=soft_label,
                                transform=train_transform if augment else test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=augment,
                                                  num_workers=num_workers, pin_memory=True)

        testset = DatasetClass(root=data_path, train=False, download=True, soft_label=soft_label,
                               transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers,
                                                 pin_memory=True)

    elif dataset == 'imagenet32':
        trainset = DatasetClass(root=data_path, train=True,
                                transform=train_transform if augment else test_transform,
                                lbl_range=(0, 100), id_range=(1, 11), soft_label=soft_label)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=augment,
                                                  num_workers=num_workers, pin_memory=True)

        testset = DatasetClass(root=data_path, train=False, transform=test_transform,
                               lbl_range=(0, 100), id_range=(1, 11), soft_label=soft_label)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers,
                                                 pin_memory=True)
    else:
        raise NotImplementedError('Dataset is not implemented.')

    return trainloader, testloader, n_classes


def get_wmloader(source, dataset, batch_size=100, num_workers=4, data_path='data/'):
    if dataset == 'mnist':
        wm_dataset = SubMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor(), source=source)
    elif dataset == 'svhn':
        wm_dataset = SubSVHN(root=data_path, split='train', download=True, transform=transforms.ToTensor(), source=source)
    else:
        raise NotImplementedError('Trigger set is not implemented.')

    return data.DataLoader(wm_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)