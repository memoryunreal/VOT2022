import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'

    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class LinearBlock(nn.Module):
    def __init__(self, in_planes, out_planes, input_sz, bias=True, batch_norm=True, relu=True):
        super().__init__()
        self.linear = nn.Linear(in_planes*input_sz*input_sz, out_planes, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)

