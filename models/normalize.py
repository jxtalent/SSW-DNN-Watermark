import torch
import math

"""
Refer to https://github.com/cleverhans-lab/entangled-watermark/blob/master/train.py
The authors use 'tf.image.per_image_standardization' to normalize the images.
"""


def standardization(x):
    """per image standardize x: (b,c,h,w)"""
    mean = torch.mean(x.view(x.size(0), -1), 1)
    std = torch.std(x.view(x.size(0), -1), 1)
    constant = torch.tensor(1.0 / math.sqrt(x.size(1) * x.size(2) * x.size(3)), device=x.device)
    adjusted_stddev = torch.maximum(std, constant)
    return (x - mean[:, None, None, None]) / (adjusted_stddev[:, None, None, None])
