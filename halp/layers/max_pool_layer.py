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
        grad_input_delta_tmp = grad_input_full - grad_input_lp
        if input_shape != grad_input_delta_tmp.shape:
            # pad grad input edges with 0
            input_w = grad_input_delta_tmp.shape[-2]
            input_h = grad_input_delta_tmp.shape[-1]
            grad_input_delta = torch.zeros(input_shape, dtype=grad_input_delta_tmp.dtype, device=grad_input_delta_tmp.device)
            grad_input_delta[:,:, 0:input_w, 0:input_h] = grad_input_delta_tmp
        else:
            grad_input_delta = grad_input_delta_tmp
        grad_input_lp = None
        grad_grad_output_lp = None
        grad_kernel_size = None
        grad_stride = None
        grad_padding = None
        return grad_input_delta, grad_input_lp, grad_grad_output_lp, grad_kernel_size, grad_stride, grad_padding


bit_center_max_pool2d = BitCenterMaxPool2DFunction.apply


class BitCenterMaxPool2D(BitCenterLayer, nn.MaxPool2d):
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

    def forward_lp(self, input):
        # Need to test do_offset mode whether gradient is updated properly
        input_lp = self.input_cache[self.cache_iter:(
            self.cache_iter + input.size(0))].cuda()
        grad_output_lp = \
            self.grad_output_cache[self.grad_cache_iter:(self.grad_cache_iter + input.size(0))].cuda()
        input_delta = input
        output = self.lp_func(input_delta, input_lp, grad_output_lp,
                              self.kernel_size, self.stride, self.padding)
        self.cache_iter = (
            self.cache_iter + input.size(0)) % self.n_train_sample
        self.grad_cache_iter = (
            self.grad_cache_iter + input.size(0)) % self.n_train_sample
        return output

    def forward_fp(self, input):
        if self.input_cache is None:
            self.input_cache = self.setup_cache(input)
            self.cache_iter = 0
        output = self.fp_func(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding)
        if self.grad_output_cache is None:
            self.grad_output_cache = self.setup_cache(output)
            self.grad_cache_iter = 0
        self.update_input_cache(input)
        return output
