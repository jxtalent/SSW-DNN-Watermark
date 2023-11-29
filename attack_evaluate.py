from scipy.stats import ttest_ind
import torch
from torch import nn
from torchvision import transforms

from datasets.dataloader import get_dataloader
from models.resnet import ResNet18, NaiveCNN, MoreNaiveCNN
from models.mobilenetv2 import MobileNetV2
from models.senet import SENet18
from models.vgg import VGG

import os

from train.trainer import test_on_watermark, test

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

from modelpool import clean_model_path, watermark_model_path


def get_model_arch(dataset, attack_type=None, model_arch=None):
    if dataset == 'fashion':
        if attack_type is None or attack_type not in ['distill', 'cross']:
            return 'vanilla'
        else:
            return 'simple'
    else:
        if attack_type is None or attack_type not in ['distill', 'cross']:
            return 'resnet18'
        elif attack_type == 'distill':
            return 'mobile'
        else:
            return model_arch


def build_model(dataset, n_class, model_arch=None):
    if dataset == 'fashion':
        if model_arch == 'vanilla':
            return MoreNaiveCNN(n_class)
        else:
            return NaiveCNN(n_class)
    else:
        if 'mobile' in model_arch:
            return MobileNetV2(n_class)
        elif 'vgg' in model_arch:
            return VGG('VGG16', n_class)
        elif 'se' in model_arch:
            return SENet18(n_class)
        else:
            return ResNet18(n_class)


def get_testloader(dataset):
    return get_dataloader(dataset)[1]


def get_pred_score(model, key):
    model.eval()
    with torch.no_grad():
        logits = model(key.images)
        prob = torch.softmax(logits, dim=1)
    return torch.gather(prob, 1, key.targets.unsqueeze(1)).squeeze().cpu().tolist()


def get_mean_and_std(x):
    return f"{np.mean(x)}+/-{np.std(x)}"


def watermark_acc(key, model):
    criterion = nn.CrossEntropyLoss()
    valid_metrics = test_on_watermark(model, criterion, key)
    return valid_metrics['acc']


def get_test_acc(model, testloader, device):
    criterion = nn.CrossEntropyLoss()
    valid_metrics = test(model, criterion, testloader, device)
    return valid_metrics['acc']


if __name__ == '__main__':
    dataset_name = 'cifar10'  # ['cifar10', 'cifar100', 'fashion']
    method = 'ssw-p'  # ['ssw-p', 'ssw-s']
    attack_type = 'retrain'  # ['retrain', 'distill', 'knockoff', 'cross', 'ftal', 'rtal', 'prune', 'quantization']
    key_num = 100           # number of images to verify
    model_arch = None   # default None, ['vgg16' or 'senet'] for cross-arch retraining.
    if attack_type == 'cross':
        assert model_arch in ['vgg16', 'senet']

    device = torch.device('cuda:0')
    num_classes = 100 if '100' in dataset_name else 10
    testloader = get_testloader(dataset_name)

    # clean model lists
    clean_nets = []
    for path in clean_model_path[dataset_name]:
        net = build_model(dataset_name, num_classes, get_model_arch(dataset_name, None, model_arch))
        net.load_state_dict(torch.load(path)['net'])
        net.to(device)
        clean_nets.append(net)

    p_values = []
    success_rates = []
    test_acc = []
    attacked_success_rates = []
    attacked_test_acc = []

    for i, path in enumerate(watermark_model_path[method][dataset_name]):
        # host model
        net = build_model(dataset_name, num_classes, get_model_arch(dataset_name, None, model_arch))
        net.load_state_dict(torch.load(path)['net'])
        net.to(device)

        test_acc.append(get_test_acc(net, testloader, device))

        # trigger set
        dirname = os.path.dirname(path)
        key = torch.load(os.path.join(dirname, 'key.pt'))
        key.to(device)

        key.images.data = key.images.data[:key_num]
        key.targets.data = key.targets.data[:key_num]

        clean_wm_acc = [watermark_acc(key, c) for c in clean_nets]
        wm_acc = watermark_acc(key, net)
        success_rates.extend([wm_acc - acc for acc in clean_wm_acc])

        # prediction scores from clean model
        clean_model_pred_score = [get_pred_score(n, key) for n in clean_nets]

        attacked_model_paths = [os.path.join(dirname, 'attack', attack_type, x, 'last.pt')
                                for x in os.listdir(os.path.join(dirname, 'attack', attack_type))
                                ]

        attacked_nets = []
        for p in attacked_model_paths:
            attacked_net = build_model(dataset_name, num_classes, get_model_arch(dataset_name, attack_type, model_arch))
            attacked_net.load_state_dict(torch.load(p)['net'])
            attacked_net.to(device)

            attacked_test_acc.append(get_test_acc(attacked_net, testloader, device))

            wm_acc = watermark_acc(key, attacked_net)
            attacked_success_rates.extend([wm_acc - acc for acc in clean_wm_acc])

            for score_clean in clean_model_pred_score:
                # prediction scores from attacked model
                score_attack = get_pred_score(attacked_net, key)
                _, p_value = ttest_ind(score_attack, score_clean, alternative='greater')
                p_values.append(p_value)

    print("Host model")
    print("test accuracy:", get_mean_and_std(np.array(test_acc)))
    print("watermark success rate:", get_mean_and_std(np.array(success_rates)))

    print("Attacked model")
    print("test accuracy:", get_mean_and_std(np.array(attacked_test_acc)))
    print("watermark success rate:", get_mean_and_std(np.array(attacked_success_rates)))
    print("watermark p-value:", get_mean_and_std(np.array(p_values)))
