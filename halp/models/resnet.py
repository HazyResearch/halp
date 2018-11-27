import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from halp.utils.utils import single_to_half_det, single_to_half_stoc, copy_layer_weights
from halp.utils.utils import void_cast_func, get_recur_attr
from halp.layers.bit_center_layer import BitCenterModule
from halp.layers.linear_layer import BitCenterLinear
from halp.layers.cross_entropy import BitCenterCrossEntropy
from halp.layers.conv_layer import BitCenterConv2D
from halp.layers.relu_layer import BitCenterReLU
from halp.layers.pool_layer import BitCenterMaxPool2D
from halp.layers.batch_norm_layer import BitCenterBatchNorm2D


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class BitCenterBasicBlock(BitCenterModule):
    expansion = 1

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 cast_func=void_cast_func,
                 n_train_sample=n_train_sample):
        super(BasicBlock, self).__init__()

        self.conv1 = BitCenterConv2D(
            in_planes,
            planes,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
            bias=False,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.bn1 = BitCenterBatchNorm2D(
            planes, cast_func=cast_func, n_train_sample=n_train_sample)

        self.relu1 = BitCenterRelu(
            cast_func=cast_func, n_train_sample=n_train_sample)

        self.conv2 = BitCenterBatchNorm2D(
            planes,
            planes,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.bn2 = BitCenterBatchNorm2D(
            planes, cast_func=cast_func, n_train_sample=n_train_sample)

        self.relu2 = BitCenterRelu(
            cast_func=cast_func, n_train_sample=n_train_sample)

        self.shortcut_conv = None
        self.shortcut_bn = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = BitCenterConv2D(
                in_planes,
                self.expansion * planes,
                kernel_size=(1, 1),
                stride=stride,
                bias=False,
                cast_func=cast_func,
                n_train_sample=n_train_sample)

            self.shortcut_bn = BitCenterBatchNorm2D(
                self.expansion * planes,
                cast_func=cast_func,
                n_train_sample=n_train_sample)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if (self.shortcut_conv is not None) and (self.shortcut_bn is not None):
            out_shortcut = self.shortcut_conv(x)
            out_shortcut = self.shortcut_bn(out_shortcut)
            out += self.shortcut(x)
        out = self.relu2(out)
        return out


class BitCenterResNet(BitCenterModule):
    def __init__(self,
                 block,
                 num_blocks,
                 num_classes=10,
                 reg_lambda=0.0,
                 dtype="bc",
                 cast_func=void_cast_func,
                 n_train_sample=n_train_sample):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.reg_lambda = reg_lambda
        self.dtype = dtype
        self.cast_func = cast_func
        self.n_train_sample = n_train_sample

        self.conv1 = BitCenterBatchNorm2D(
            3,
            64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.bn1 = BitCenterBatchNorm2D(
            64, cast_func=cast_func, n_train_sample=n_train_sample)

        self.relu1 = BitCenterRelu(
            cast_func=cast_func, n_train_sample=n_train_sample)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avg_pool = BitCenterAvgPool2D(
            kernel_size=(4, 4),
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.linear = BitCenterLinear(
            512 * block.expansion,
            num_classes,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return BitCenterSequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
