import torch

import torch.nn as nn
from datasets.dataloader import get_wmloader


class Key(nn.Module):
    def __init__(self, key_num, key_target, channels, device, data_path='data/'):
        super().__init__()
        wm_dataset = 'mnist' if channels == 1 else 'svhn'
        source = 8 if channels == 1 else 9

        wm_loader = get_wmloader(source, wm_dataset, data_path=data_path)
        data = torch.cat([d for d, _ in wm_loader], dim=0)

        self.images = nn.Parameter(data[:key_num])

        targets = torch.ones(len(self.images), device=device).long() * key_target
        self.register_buffer('targets', targets)
