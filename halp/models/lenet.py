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
from halp.layers.max_pool_layer import BitCenterMaxPool2D


class LeNet_PyTorch(nn.Module):
    def __init__(self):
        super(LeNet_PyTorch, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet(BitCenterModule):
    def __init__(self, dtype="bc", cast_func=void_cast_func, n_train_sample=1):
        super(LeNet, self).__init__()
        self.cast_func = cast_func
        self.n_train_sample = n_train_sample
        self.dtype = dtype
        # setup layers
        self.conv1 = BitCenterConv2D(
            in_channels=3,
            out_channels=6,
            kernel_size=(5, 5),
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.relu1 = BitCenterReLU(
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.max_pool1 = BitCenterMaxPool2D(
            kernel_size=(2, 2),
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.conv2 = BitCenterConv2D(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.relu2 = BitCenterReLU(
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.max_pool2 = BitCenterMaxPool2D(
            kernel_size=(2, 2),
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.fc1 = BitCenterLinear(
            in_features=16 * 5 * 5,
            out_features=120,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.relu3 = BitCenterReLU(
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.fc2 = BitCenterLinear(
            in_features=120,
            out_features=84,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.relu4 = BitCenterReLU(
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        self.fc3 = BitCenterLinear(
            in_features=84,
            out_features=10,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.criterion = BitCenterCrossEntropy(
            cast_func=cast_func, n_train_sample=n_train_sample)

        if dtype == "bc":
            pass
        elif (dtype == "fp") or (dtype == "lp"):
            self.conv1 = copy_layer_weights(self.conv1, nn.Conv2d(3, 6, 5))
            self.conv2 = copy_layer_weights(self.conv2, nn.Conv2d(6, 16, 5))
            self.fc1   = copy_layer_weights(self.fc1, nn.Linear(16*5*5, 120))
            self.fc2   = copy_layer_weights(self.fc2, nn.Linear(120, 84))
            

            # def update_grad_output_cache1(self, input, output):
            #     # use duplicated self to adapt to the pytorch API requirement
            #     # as this is a class member function.
            #     # Specific layer might need to update this function. This is
            #     # because the returned gradient is not in the order as shown
            #     # in the Python API, e.g. the linear layer
            #     print("check fp linear output grad ", torch.sum(output[0]**2).item())
            #     print("check fp linear input grad ", torch.sum(input[1]**2).item())

            # def check_fw1(self, input, output):
            #     # print("type check ", len(input), len(output), input[0].shape)
            #     print("check linear forward non zero ", torch.sum(input[0]**2).item(), input[0].numel())
            # self.fc2.register_forward_hook(check_fw1)
            # self.fc2.register_backward_hook(update_grad_output_cache1)


            self.fc3   = copy_layer_weights(self.fc3, nn.Linear(84, 10))
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()

            # def update_grad_output_cache(self, input, output):
            #     # use duplicated self to adapt to the pytorch API requirement
            #     # as this is a class member function.
            #     # Specific layer might need to update this function. This is
            #     # because the returned gradient is not in the order as shown
            #     # in the Python API, e.g. the linear layer
            #     print("check fp relu output grad ", torch.sum(output[0]**2).item())
            #     print("check fp relu input grad ", torch.sum(input[0]**2).item(), torch.sum(input[0] != 0).item())

            # def check_fw(self, input, output):
            #     print("type check ", len(input), len(output), input[0].shape)
            #     print("check relu forward non zero ", torch.sum(input[0]**2).item(), input[0].numel(), torch.sum((input[0] <= 0)).item(), torch.sum((output[0] != 0)).item())

            # self.relu4.register_backward_hook(update_grad_output_cache)
            # self.relu4.register_forward_hook(check_fw)

            self.max_pool1 = nn.MaxPool2d(kernel_size=2)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2)
            self.criterion = nn.CrossEntropyLoss(size_average=True)

            if dtype == "lp":
                if self.cast_func == void_cast_func:
                    pass
                else:
                    for child in self.children():
                        child.half()
        else:
            raise Exception(dtype + " is not supported in LeNet!")

    def forward(self, x, y, test=False):
        # print("input ", torch.sum(x**2), torch.sum(y**2))
        # print("weight ", torch.sum(self.conv1.weight**2), torch.sum(self.conv1.bias**2))

        # tmp = nn.Conv2d(3, 6, 5).cuda()
        # tmp = copy_layer_weights(self.conv1, tmp)
        # tmp_out = tmp(x)
        # x[:] = 1.0
        # self.conv1.weight.data.zero_()
        # self.conv1.bias.data.zero_()
        # print("ckpt1.1 ", x[0, 0, 0, :], self.conv1(x)[0,0,0,:], self.conv1.__class__.__name__, tmp.__class__.__name__)
        # print("ckpt1.2 ", self.conv1.weight.shape, 
        #     self.conv1.bias.shape, x.dtype, self.conv1.bias)
        # print("ckpt1.3 ", torch.sum(x**2), torch.sum(self.conv1(x)**2))
        # print("ckpt 1.4 ", torch.sum(self.conv1.weight), torch.sum(self.conv1.bias), self.conv1.stride,
        #                       self.conv1.padding, self.conv1.dilation, self.conv1.groups)



        out = self.relu1(self.conv1(x))
        # out = self.conv1(x)


        # print("forward residual ", torch.sum(out**2).item())



        out = self.max_pool1(out)

        # print("forward residual ", torch.sum(out**2).item())


        out = self.relu2(self.conv2(out))
        # out = self.conv2(out)


        # print("forward residual ", torch.sum(out**2).item())


        out = self.max_pool2(out)

        # print("forward residual ", torch.sum(out**2).item())


        # print("ckpt1 ", torch.sum(out**2))


        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        # out = self.fc1(out)


        # print("forward residual ", torch.sum(out**2).item())


        out = self.relu4(self.fc2(out))
        # out = self.fc2(out)

        # print("relu in ", torch.sum(out**2).item())

        # out = self.relu4(out)

        # print("relu out ", torch.sum(out**2).item())

        # print("forward residual ", torch.sum(out**2).item())


        out = self.fc3(out)

        # print("forward residual ", torch.sum(out**2).item())


        if test:
            return out
        else:
            # print("output in shape train ", out.shape, y.shape, self.criterion.__class__.__name__)
            self.loss = self.criterion(out, y)
            # print("ckpt -1 ", self.loss**2)
            # exit(0)
            # print("test addition done")
            return self.loss

    def check_layer_status(self, do_offset=True):
        assert self.conv1.do_offset == do_offset
        assert self.relu1.do_offset == do_offset
        assert self.max_pool1.do_offset == do_offset
        assert self.conv2.do_offset == do_offset
        assert self.relu2.do_offset == do_offset
        assert self.max_pool2.do_offset == do_offset
        assert self.fc1.do_offset == do_offset
        assert self.relu3.do_offset == do_offset
        assert self.fc2.do_offset == do_offset
        assert self.relu4.do_offset == do_offset
        assert self.fc3.do_offset == do_offset
        assert self.criterion.do_offset == do_offset


    def predict(self, x):
        # print("input x in shape train ", x.shape)
        output = self.forward(x, y=None, test=True)
        pred = output.data.cpu().numpy().argmax(axis=1)
        return pred, output

