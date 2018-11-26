import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from halp.layers.bit_center_layer import BitCenterLayer
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class BitCenterMaxPool2DFunction(Function):
    @staticmethod
    def forward(ctx,
                input_delta,
                input_lp,
                grad_output_lp,
                kernel_size,
                stride=None,
                padding=0):
        input_full = input_lp + input_delta
        output_full, indices_full = F.max_pool2d(
            input_full,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True)
        # The following line is just for simulation, the output and indices can be cached
        output_lp, indices_lp = F.max_pool2d(
            input_lp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True)
        ctx.save_for_backward(grad_output_lp, indices_full, indices_lp)
        ctx.hyperparam = (kernel_size, stride, padding, input_full.shape)
        if type(padding) is list:
            raise Exception("tuple based padding ")
        return output_full - output_lp

    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, stride, padding, input_shape = ctx.hyperparam
        grad_output_lp, indices_full, indices_lp = ctx.saved_tensors
        grad_output_full = grad_output + grad_output_lp
        grad_input_full = F.max_unpool2d(
            grad_output_full,
            indices_full,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_size=input_shape)
        grad_input_lp = F.max_unpool2d(
            grad_output_lp,
            indices_lp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_size=input_shape)
        grad_input_delta = grad_input_full - grad_input_lp

        # note here pytorch max pool layer set stride = kernel size if not specified
        if (stride != kernel_size) or (padding is not 0):
            raise Exception("stride and padding are not fully supported yet!")
        grad_input_lp = None
        grad_grad_output_lp = None
        grad_kernel_size = None
        grad_stride = None
        grad_padding = None
        return grad_input_delta, grad_input_lp, grad_grad_output_lp, grad_kernel_size, grad_stride, grad_padding


class BitCenterAvgPool2DFunction(Function):
    @staticmethod
    def forward(ctx,
                input_delta,
                input_lp,
                grad_output_lp,
                kernel_size,
                stride=None,
                padding=0):
        input_full = input_lp + input_delta
        output_full = F.avg_pool2d(
            input_full,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        # The following line is just for simulation, the output and indices can be cached
        output_lp = F.avg_pool2d(
            input_lp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        ctx.save_for_backward(grad_output_lp)
        ctx.hyperparam = (kernel_size, stride, padding, input_full.shape)
        if type(padding) is list:
            raise Exception("tuple based padding ")
        return output_full - output_lp

    @staticmethod
    def backward(ctx, grad_output):
        # note here pytorch avg pool layer set stride = kernel size if not specified
        if (stride != kernel_size):
            raise Exception("stride is only supported when it is equal to kernel size!")
        kernel_size, stride, padding, input_shape = ctx.hyperparam
        grad_output_lp = ctx.saved_tensors
        grad_output_full = grad_output + grad_output_lp

        shape = list(grad_output.shape)
        grad_input = grad_output.view(*shape, 1).contiguous()\
            .expand(*shape, kernel_size[1]).contiguous()\
            .view(*shape[:-1], shape[-1]*kernel_size[-1])
        shape = list(grad_input.shape)
        grad_input = grad_input.view(*shape[:-1], 1, shape[-1]).contiguous() \
            .expand(*shape[:-1], kernel_size[0], shape[-1]).contiguous()\
            .view(*shape[:-2], shape[-2] * kernel_size[0], shape[-1])
        if list(grad_input.shape) != list(input_shape):
            grad_input_tmp = grad_input
            grad_input = np.zeros(input_shape)
            grad_input[:, :, :grad_input_tmp.size(2), :grad_input_tmp.size(3)] = grad_input_tmp
        grad_input_delta = grad_input_full - grad_input_lp

        grad_input_lp = None
        grad_grad_output_lp = None
        grad_kernel_size = None
        grad_stride = None
        grad_padding = None
        return grad_input_delta, grad_input_lp, grad_grad_output_lp, grad_kernel_size, grad_stride, grad_padding


bit_center_max_pool2d = BitCenterMaxPool2DFunction.apply

bit_center_avg_pool2d = BitCenterAvgPool2DFunction.apply


class BitCenterPool2D(BitCenterLayer):
    def forward_lp(self, input):
        input_lp, grad_output_lp = self.get_input_cache_grad_cache(input)
        input_delta = input
        output = self.lp_func(input_delta, input_lp, grad_output_lp,
                              self.kernel_size, self.stride, self.padding)
        self.increment_cache_iter(input)
        return output

    def forward_fp(self, input):
        self.check_or_setup_input_cache(input)
        output = self.fp_func(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding)
        self.check_or_setup_grad_cache(output)
        self.update_input_cache(input)
        return output


class BitCenterMaxPool2D(BitCenterPool2D, nn.MaxPool2d):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterLayer.__init__(
            self,
            fp_functional=F.max_pool2d,
            lp_functional=bit_center_max_pool2d,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        nn.MaxPool2d.__init__(
            self, kernel_size=kernel_size, stride=stride, padding=padding)
        self.register_backward_hook(self.update_grad_output_cache)


class BitCenterAvgPool2D(BitCenterPool2D, nn.AvgPool2d):
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterLayer.__init__(
            self,
            fp_functional=F.avg_pool2d,
            lp_functional=bit_center_avg_pool2d,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        nn.AvgPool2d.__init__(
            self, kernel_size=kernel_size, stride=stride, padding=padding)
        self.register_backward_hook(self.update_grad_output_cache)





