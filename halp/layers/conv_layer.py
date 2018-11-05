import torch
import torch.nn as nn
from torch.nn import Conv2d
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


class BitCenterConv2DFunction(Function):
    @staticmethod
    def forward(ctx,
                input_delta,
                input_lp,
                output_grad_lp,
                weight_delta,
                weight_lp,
                bias_delta=None,
                bias_lp=None,
                stride=1,
                padding=0,
                dilation=1,
                group):
        # suffix lp means the lp version of the offset tensors
        # suffix delta means the real low precision part of the model representation
        # output_grad_lp is only for backward function, but we need to keep it in ctx
        # for backward function to access it.
        ctx.save_for_backward(input_lp, input_delta, output_grad_lp, weight_lp,
                              weight_delta, bias_lp, bias_delta, stride,
                              padding, dilation, group)

        conv2d = lambda input, weight, bias: F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        output = conv2d(input_delta, weight_lp, bias_delta) \
         + conv2d(input_lp + input_delta, weight_delta, None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
    	Given O = X * F, we have
    	\par L / \par F = X * \par L / \par O
    	\par L / \par X = \par L / \par O * flip_180(w)
    	The shape of the tensor: N, C_{in}, H_{in}, W_{in}
    	'''
        input_lp, input_delta, output_grad_lp, \
            weight_lp, weight_delta, bias_lp, bias_delta, \
            stride, padding, dilation, group = ctx.saved_tensors
        conv2d = lambda input, weight, bias: F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        grad_input_lp = None
        grad_input_delta = \
         conv2d(grad_output, torch.flip((weight_lp + weight_delta), [2, 3]), None) \
         + conv2d(output_grad_lp, torch.flip(weight_delta, [2, 3]), None)
        grad_output_grad_lp = None  # this dummy to adapt to pytorch API
        grad_weight_lp = None
        grad_weight_delta = conv2d(input_lp + input_delta, grad_output, None) \
         + conv2d(input_delta, output_grad_lp, None)
        grad_bias_lp = None
        if (bias_lp is not None) and (bias_delta is not None):
            grad_bias_delta = grad_output.sum(dim=[0, 2, 3])
        else:
            grad_bias_delta = None
        grad_stride, grad_padding, grad_dilation, grad_group = None, None, None, None
        return grad_input_delta, grad_input_lp, grad_output_grad_lp, \
            grad_weight_delta, grad_weight_lp, grad_bias_delta, grad_bias_lp, \
            grad_stride, grad_padding, grad_dilation, grad_group


bit_center_conv2d = BitCenterConv2DFunction.apply


class BitCenterConv2D(BitCenterLayer, Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=bias,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterLayer.__init__(
            self,
            fp_functional=F.conv2d,
            lp_functional=bit_center_linear,
            bias=bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # weight_delta is the delta tensor in the algorithm while weight_lp is the cached
        # lp version of weight offset
        self.setup_bit_center_vars()
        self.cuda()
        self.reset_parameters_bit_center()
        self.register_backward_hook(self.update_grad_output_cache)

    def update_grad_output_cache(self, self1, input, output):
        # use duplicated self to adapt to the pytorch API requirement
        # as this is a class member function
        if self.do_offset:
            self.grad_output_cache[self.grad_cache_iter:min(
                self.grad_cache_iter +
                output[0].size()[0], self.n_train_sample)].data.copy_(
                    self.cast_func(output[0].cpu()))
            self.grad_cache_iter = (
                self.grad_cache_iter + output[0].size(0)) % self.n_train_sample
            # we use the following variable only for test purpose, we want to be able to access
            # the gradeint value wrt input in the outside world. For lp mode, it is grad_input_delta
            # for fp mode, it is grad_input
            # TODO: update if pytorch stable version fixes this:
            # The following branch is due to the layer specific behavior of
            # input argument to the backward hook.
            # Here we hard code the order of tensor in the input list (this is layer specific)
            if self.bias is not None:
                self.input_grad_for_test = input[1]
            else:
                self.input_grad_for_test = input[0]
        else:
            self.input_grad_for_test = input[0]
