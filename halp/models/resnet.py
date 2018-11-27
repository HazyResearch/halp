import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from halp.utils.utils import single_to_half_det, single_to_half_stoc
from halp.utils.utils import copy_layer_weights, copy_module_weights
from halp.utils.utils import void_cast_func, get_recur_attr
from halp.layers.bit_center_layer import BitCenterModule
from halp.layers.bit_center_layer import BitCenterSequential
from halp.layers.linear_layer import BitCenterLinear
from halp.layers.cross_entropy import BitCenterCrossEntropy
from halp.layers.conv_layer import BitCenterConv2D
from halp.layers.relu_layer import BitCenterReLU
from halp.layers.pool_layer import BitCenterAvgPool2D
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

        # print("ckpt 1", torch.sum(x**2))


        out = F.relu(self.bn1(self.conv1(x)))

        # print("ckpt 2", torch.sum(x**2))


        out = self.bn2(self.conv2(out))

        # print("ckpt 3", torch.sum(x**2))


        out += self.shortcut(x)

        # print("ckpt 4", torch.sum(x**2))


        out = F.relu(out)
        return out


class ResNet_PyTorch(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_PyTorch, self).__init__()
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


# def ResNet18_PyTorch():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


class BitCenterBasicBlock(BitCenterModule):
    expansion = 1

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        super(BitCenterBasicBlock, self).__init__()

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

        self.relu1 = BitCenterReLU(
            cast_func=cast_func, n_train_sample=n_train_sample)

        self.conv2 = BitCenterConv2D(
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

        self.relu2 = BitCenterReLU(
            cast_func=cast_func, n_train_sample=n_train_sample)

        self.shortcut = BitCenterSequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = BitCenterSequential(
                BitCenterConv2D(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=(1, 1),
                    stride=stride,
                    bias=False,
                    cast_func=cast_func,
                    n_train_sample=n_train_sample),
                BitCenterBatchNorm2D(
                    self.expansion * planes,
                    cast_func=cast_func,
                    n_train_sample=n_train_sample))

    def forward(self, x):

        # print("ckpt 1", torch.sum(x**2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # print("ckpt 2", torch.sum(x**2))


        out = self.conv2(out)
        out = self.bn2(out)

        # print("ckpt 3", torch.sum(x**2))


        out += self.shortcut(x)

        # print("ckpt 4", torch.sum(x**2))


        out = self.relu2(out)
        return out


class ResNet(BitCenterModule):
    def __init__(self,
                 block,
                 num_blocks,
                 num_classes=10,
                 reg_lambda=0.0,
                 dtype="bc",
                 cast_func=void_cast_func,
                 n_train_sample=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.reg_lambda = reg_lambda
        self.dtype = dtype
        self.cast_func = cast_func
        self.n_train_sample = n_train_sample

        self.conv1 = BitCenterConv2D(
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

        self.relu1 = BitCenterReLU(
            cast_func=cast_func, n_train_sample=n_train_sample)

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

        self.criterion = BitCenterCrossEntropy(
            cast_func=cast_func, n_train_sample=n_train_sample)

        if dtype == "bc":
            pass
        elif (dtype == "fp") or (dtype == "lp"):
            # for fp and lp models, we use the origianl pytorch modules
            # reset initial inplanes
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()

            self.layer1 = ResNet_PyTorch._make_layer(
                self, BasicBlock, 64, num_blocks[0], stride=1)
            self.layer2 = ResNet_PyTorch._make_layer(
                self, BasicBlock, 128, num_blocks[1], stride=2)
            self.layer3 = ResNet_PyTorch._make_layer(
                self, BasicBlock, 256, num_blocks[2], stride=2)
            self.layer4 = ResNet_PyTorch._make_layer(
                self, BasicBlock, 512, num_blocks[3], stride=2)

            self.avg_pool = nn.AvgPool2d(kernel_size=(4, 4))
            self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)
            
            self.criterion = nn.CrossEntropyLoss(size_average=True)
            if dtype == "lp":
                if self.cast_func == void_cast_func:
                    pass
                else:
                    for child in self.children():
                        child.half()
        else:
            raise Exception(dtype + " is not supported in LeNet!")

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.cast_func, self.n_train_sample))
            self.in_planes = planes * block.expansion
        return BitCenterSequential(*layers)

    def forward(self, x, y, test=False):
        out = self.relu1(self.bn1(self.conv1(x)))

        # print(self.dtype, "overall ckpt 1", torch.sum(out**2))

        out = self.layer1(out)

        # print(self.dtype, "overall ckpt 2", torch.sum(out**2))

        out = self.layer2(out)

        # print(self.dtype, "overall ckpt 3", torch.sum(out**2))

        out = self.layer3(out)

        # print(self.dtype, "overall ckpt 4", torch.sum(out**2))

        out = self.layer4(out)

        # print(self.dtype, "overall ckpt 5", torch.sum(out**2))

        out = self.avg_pool(out)

        # print(self.dtype, "overall ckpt 6", torch.sum(out**2))

        out = out.view(out.size(0), -1)

        # print(self.dtype, "overall ckpt 7", torch.sum(out**2))

        out = self.linear(out)

        # print(self.dtype, "overall ckpt 7", torch.sum(out**2))

        self.output = out
        if test:
            return out
        else:
            self.loss = self.criterion(out, y)
            if isinstance(self.criterion, BitCenterCrossEntropy) \
                and self.criterion.do_offset == False:
                # this is for the case where we want to get full output
                # in the do_offset = False mode.
                self.output = self.output + self.criterion.input_lp
            return self.loss

    def predict(self, x):
        output = self.forward(x, y=None, test=True)
        pred = output.data.cpu().numpy().argmax(axis=1)
        return pred, output

    def check_layer_status(self, do_offset=True):
        pass
