import copy

import numpy as np
import torch
from torch import nn


def weight_prune(model, pruning_perc):
    new_model = copy.deepcopy(model)
    if pruning_perc == 0:
        return new_model

    all_weights = np.concatenate([p.abs().data.cpu().numpy().reshape(-1) for p in new_model.parameters()
                                  if len(p.data.size()) != 1])

    threshold = np.percentile(all_weights, pruning_perc)
    for p in new_model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())
    return new_model


def quantization(param, bits):
    quantata = int(np.math.pow(2, bits))
    min_weight, max_weight = param.data.min(), param.data.max()
    qranges = torch.linspace(min_weight, max_weight, quantata)

    ones = torch.ones_like(param.data)
    zeros = torch.zeros_like(param.data)
    for i in range(len(qranges) - 1):
        t1 = torch.where(param.data > qranges[i], zeros, ones)
        t2 = torch.where(param.data < qranges[i + 1], zeros, ones)
        t3 = torch.where((t1 + t2) == 0, ones * (qranges[i] + qranges[i + 1]) / 2, zeros)
        t4 = torch.where((t1 + t2) == 0, zeros, ones)

        param.data = t4 * param.data + t3
    return param


def re_initializer_layer(model, num_classes, device, layer=None):
    """remove the last layer and add a new one"""
    if hasattr(model, 'linear'):
        private_key = model.linear
    else:
        private_key = model.fc2
    indim = private_key.in_features
    if layer:
        if hasattr(model, 'linear'):
            model.linear = layer
        else:
            model.fc2 = layer
    else:
        if hasattr(model, 'linear'):
            model.linear = nn.Linear(indim, num_classes).to(device)
        else:
            model.fc2 = nn.Linear(indim, num_classes).to(device)
    return model, private_key
