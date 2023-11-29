import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image
import pickle
import os


class SubMNIST(datasets.MNIST):
    def __init__(self, root, train, download, transform=None, source=0):
        super(SubMNIST, self).__init__(root=root, train=train, download=download, transform=transform)
        data = self.data[self.targets == source]
        targets = self.targets[self.targets == source]
        self.data = data
        self.targets = targets


class SubSVHN(datasets.SVHN):
    def __init__(self, root, split, download, transform=None, source=0):
        super(SubSVHN, self).__init__(root=root, split=split, download=download, transform=transform)
        data = self.data[self.labels == source]
        targets = self.labels[self.labels == source]
        self.data = data
        self.labels = targets


class SoftLabelCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download, soft_label, transform=None):
        super(SoftLabelCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: np(10000, 32, 32, 3) unit8,  self.soft_label: tensor(10000, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SoftLabelCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train, download, soft_label, transform=None):
        super(SoftLabelCIFAR100, self).__init__(root=root, train=train, download=download, transform=transform)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: np(10000, 32, 32, 3) unit8,  self.soft_label: tensor(10000, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class SoftLabelMNIST(datasets.MNIST):
    def __init__(self, root, train, download, soft_label, transform=None):
        super(SoftLabelMNIST, self).__init__(root=root, train=train, download=download, transform=transform)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: tensor (60000, 28, 28) unit8,  self.targets: tensor (6w, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SoftLabelFashion(datasets.FashionMNIST):
    def __init__(self, root, train, download, soft_label, transform=None):
        super(SoftLabelFashion, self).__init__(root=root, train=train, download=download, transform=transform)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: tensor (60000, 28, 28) unit8,  self.targets: tensor (6w, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SoftLabelKMNIST(datasets.KMNIST):
    def __init__(self, root, train, download, soft_label, transform=None):
        super(SoftLabelKMNIST, self).__init__(root=root, train=train, download=download, transform=transform)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: tensor (60000, 28, 28) unit8,  self.targets: tensor (6w, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SoftLabelEMNIST(datasets.EMNIST):
    def __init__(self, root, train, download, soft_label, transform=None, split='letters'):
        super(SoftLabelEMNIST, self).__init__(root=root, train=train, download=download, transform=transform, split=split)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: tensor (60000, 28, 28) unit8,  self.targets: tensor (6w, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


class IMAGENET32(data.Dataset):
    def __init__(self, root, train=True,
                 transform=None, lbl_range=(0, 1000), id_range=(1, 11)):
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.lbl_range = lbl_range
        self.id_range = id_range

        self.data = []
        self.targets = []
        if self.train:
            for idx in range(id_range[0], id_range[1]):
                if lbl_range[1] == 1002:
                    x, y = unpickle(os.path.join(self.root, 'Imagenet32_train/train_batch_py2_') + str(idx))
                else:
                    x, y = self.loaddata(os.path.join(self.root, 'Imagenet32_train/train_data_batch_') + str(idx))
                    # each x: #samples, samples x 3072 (R, G, B)
                if lbl_range[1] == 1001:
                    # dump data with protocol 2
                    with open(os.path.join(self.root, 'Imagenet32_train/train_batch_py2_') + str(idx), 'wb') as fo:
                        pickle.dump((x, y), fo, 2)

                self.data.append(x)
                self.targets.extend(y)
                # print("loaded:", idx)

        else:
            x, y = self.loaddata(os.path.join(self.root, 'Imagenet32_val/val_data'))
            self.data.append(x)
            self.targets.extend(y)

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

        self.targets = [y - 1 for y in self.targets]

        if lbl_range[0] > 0 or lbl_range[1] < 1000:
            _data = self.data
            _targets = self.targets
            self.data = []
            self.targets = []

            for i in range(_data.shape[0]):
                if _targets[i] >= lbl_range[0] and _targets[i] < lbl_range[1]:
                    self.data.append(_data[i])
                    self.targets.append(_targets[i])

            self.data = np.stack(self.data)

    def loaddata(self, path):
        d = unpickle(path)
        return d['data'], d['labels']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class SoftIMAGENET32(IMAGENET32):
    def __init__(self, root, train, transform, lbl_range, id_range, soft_label):
        super(SoftIMAGENET32, self).__init__(root, train, transform, lbl_range, id_range)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: np(10000, 32, 32, 3) unit8,  self.soft_label: tensor(10000, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class GrayCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download, transform=None):
        super(GrayCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)
        self.data = self.data[:, :28, :28, 0]

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class SoftGrayCIFAR10(GrayCIFAR10):
    def __init__(self, root, train, download, transform, soft_label):
        super(SoftGrayCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)
        self.soft_label = soft_label

    def __getitem__(self, index):
        # self.data: np(10000, 32, 32, 3) unit8,  self.soft_label: tensor(10000, 10) cpu
        img, target = self.data[index], self.soft_label[index]
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
