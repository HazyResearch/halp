import torch
import torch.nn as nn
from math import floor
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

# helper functions for backward computation of bit center conv 2d:
# we define these function to get gradient wrt input and weights
get_grad_input = lambda grad_out_reshape, weight: \
    F.conv_transpose2d(grad_out_reshape, weight, bias=None)
get_grad_weight = lambda input_unf, grad_output_reshape: \
    torch.sum(torch.bmm(input_unf, grad_output_reshape), dim=0)


class BitCenterConv2DFunction(Function):
    """
    This class does forward and backward computation for bit center 
    2D convolution. We document the forward backward computation math
    here for normal conv2d. The math for bit center Conv2d can be derived
    by decomposing the input and weights into full precision offset and 
    low precision delta.
    Given input X in the shape of (b, n_i, w_i, h_i)
    and filter W in the shape of (n_o, n_i, w_k, h_k)
    For simplicity we write out the math with 0 padding and stride 1 
    in the following:
    we transform the input tensor X to a matrix X_L in the shape of 
    (b, w_o * h_o, n_i * w_k * h_k). W is transformed into W_L in the shape of
    (n_i * w_k * h_k, n_o).
    In the forward pass,
    \tilde{X} = matmul(X_L, W_L), where \tilde{X} is in the shape of
    (b, w_o * h_o, n_o)
    In the backward pass,
    \par L / \par W_L = matmul(X_L^T, \par L / \par \tilde{X}_L)
    \par L / \par X_L = matmul(\par L / \par \tilde{X}_L, W_L^T)
    Note here \par L / \par X_L can be directly done using deconv operations
    We opt to use fold and unfold function to explicitly do 
    \par L / \par W_L = matmul(X_L^T, \par L / \par \tilde{X}_L)
    because we the current deconv (conv_transpose2d) API has some flexibility issue.
    Note \par L / \par W_L, and \par L / \par X_L are in the shape of 
    () and () respectively. We need properly reshape to return the gradient
    """
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
                groups=1):
        # suffix lp means the lp version of the offset tensors
        # suffix delta means the real low precision part of the model representation
        # output_grad_lp is only for backward function, but we need to keep it in ctx
        # for backward function to access it.
        # TODO: extend to accommodate different stride, padding, dilation, groups
        assert (stride, padding, dilation, groups) == (1, 0, 1, 1) 
        batch_size = input_lp.size(0)
        kernel_size = np.array(weight_lp.size()[-2:]).astype(np.int)
        input_size = np.array(input_lp.size()[-2:]).astype(np.int)
        output_size = np.floor((input_size + 2 * padding - dilation *
                                (kernel_size - 1) - 1) / stride + 1).astype(np.int)
        kernel_size = kernel_size.tolist()
        input_size = input_size.tolist()
        output_size = output_size.tolist()
        input_lp_unf = F.unfold(input_lp, kernel_size)
        input_delta_unf = F.unfold(input_delta, kernel_size)
        ctx.save_for_backward(input_lp_unf, input_delta_unf, output_grad_lp,
                              weight_lp, weight_delta, bias_lp, bias_delta)
        ctx.hyperparam = (stride, padding, dilation, groups)
        conv2d = lambda input_unf, weight: \
            input_unf.transpose(1, 2).matmul(
            weight.permute(1, 2, 3, 0).view(-1, weight.size(0)))
        output = conv2d(input_delta_unf, weight_lp) \
            + conv2d(input_lp_unf + input_delta_unf, weight_delta)
        # print(output.size(), output_size, kernel_size, input_lp.size(), weight_lp.size())
        channel_out = weight_lp.size(0)
        output = output.transpose(1, 2).view(batch_size, channel_out, *output_size)
        if bias_delta is not None:
            output = output + bias_delta.view(1, -1, 1, 1).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''
        In this function, suffix represent the results from torch unfold style im2col op
    	'''
        # TODO extend to accommodate more configs for stride, padding, dilation, groups
        input_lp_unf, input_delta_unf, output_grad_lp, \
            weight_lp, weight_delta, bias_lp, bias_delta = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.hyperparam
        assert (stride, padding, dilation, groups) == (1, 0, 1, 1) 
        batch_size, channel_out, w_out, h_out = list(grad_output.size())
        channel_in = weight_lp.size(1)
        kernel_size = weight_lp.size()[-2:] 
        assert channel_out == weight_lp.size(0)
        # reshape output grad for further computation
        grad_output_reshape = grad_output.permute(0, 2, 3, 1).view(batch_size, -1, channel_out)
        output_grad_lp_reshape = output_grad_lp.permute(0, 2, 3, 1).view(batch_size, -1, channel_out)
        grad_input_lp = None
        grad_input_delta = \
            get_grad_input(grad_output, (weight_lp + weight_delta)) \
            + get_grad_input(output_grad_lp, weight_delta)
        grad_output_grad_lp = None  # this dummy to adapt to pytorch API
        grad_weight_lp = None
        grad_weight_delta = \
            get_grad_weight(input_lp_unf + input_delta_unf, grad_output_reshape) \
            + get_grad_weight(input_delta_unf, output_grad_lp_reshape)
        grad_weight_delta = \
            grad_weight_delta.view(channel_in, *kernel_size, channel_out).permute(3, 0, 1, 2)
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
                 bias=False,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterLayer.__init__(
            self,
            fp_functional=F.conv2d,
            lp_functional=bit_center_conv2d,
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
        # as this is a class member function.
        # TODO: consider merge this function into the base BitCenterLayer
        # class. Currently it is not doable because of the behavior described
        # in BitCenterLinear layer.
        if self.do_offset:
            self.grad_output_cache[self.grad_cache_iter:min(
                self.grad_cache_iter +
                output[0].size()[0], self.n_train_sample)].data.copy_(
                    self.cast_func(output[0].cpu()))
            self.grad_cache_iter = (
                self.grad_cache_iter + output[0].size(0)) % self.n_train_sample
        self.input_grad_for_test = input[0]
