import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc, void_func
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class BitCenterModule(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def set_mode(self, do_offset, cache_iter=0):
        self.do_offset = do_offset
        self.cache_iter = cache_iter
        for child in self.children():
            if isinstance(child, BitCenterModule):
                child.set_mode(do_offset, cache_iter)
            else:
                logger.warning("None bit centering module " \
                               + child.__class__.__name__)

class BitCenterModuleList(BitCenterModule, nn.ModuleList):
    def __init__(self, modules=None):
        BitCenterModule.__init__(self)
        nn.ModuleList.__init__(self, modules)


class BitCenterLayer(BitCenterModule):
    def __init__(self,
                 fp_functional=void_func,
                 lp_functional=void_func,
                 bias=True,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterModule.__init__(self)
        self.cast_func = cast_func
        # register the fp and lp forward function
        self.fp_func = fp_functional
        self.lp_func = lp_functional
        # input cache
        self.input_cache = None
        self.grad_output_cache = None
        self.cache_iter = 0
        self.grad_cache_iter = 0
        self.n_train_sample = n_train_sample
        # starting from fp mode
        self.set_mode(do_offset=True)

    def setup_bit_center_vars(self):
        self.weight_delta = Parameter(
            self.cast_func(self.weight.data), requires_grad=True)
        self.weight_lp = Parameter(
            self.cast_func(self.weight.data), requires_grad=False)
        self.set_mode(do_offset=True)
        if self.bias is not None:
            self.bias_delta = Parameter(
                self.cast_func(self.bias), requires_grad=True)
            self.bias_lp = Parameter(
                self.cast_func(self.bias.data), requires_grad=False)
        else:
            self.register_parameter('bias_delta', None)
            self.register_parameter('bias_lp', None)

    def reset_parameters_bit_center(self):
        init.zeros_(self.weight_delta)
        if self.bias is not None:
            init.zeros_(self.bias_delta)

    # def set_mode(self, do_offset, cache_iter=0):
    #     self.do_offset = do_offset
    #     self.cache_iter = cache_iter

    def setup_cache(self, input):
        # the cache is set up when the first minibatch forward is done.
        # here we assume the first dimension of input blob indicates the size of minibatch
        if len(list(input.size())) == 0:
            # this is the scalar output case
            # loss layers need this to be consistent with the setup of bit center layers
            cache_shape = [1, 1]
        else:
            cache_shape = list(input.size())
        cache_shape[0] = self.n_train_sample
        cache = self.cast_func(
            Variable(torch.zeros(cache_shape).type(input.dtype))).cpu()
        return cache

    def update_grad_output_cache(self, self1, input, output):
        pass

    def update_input_cache(self, input):
        if self.do_offset:
            self.input_cache[self.cache_iter:min(
                self.cache_iter +
                input.size()[0], self.n_train_sample)].data.copy_(
                    self.cast_func(input.cpu()))
            self.cache_iter = (
                self.cache_iter + input.size(0)) % self.n_train_sample

    def forward_fp(self, input):
        if self.input_cache is None:
            self.input_cache = self.setup_cache(input)
            self.cache_iter = 0

        # print("fp forward input ", torch.sum(input**2))

        output = self.fp_func(input, self.weight, self.bias)
        if self.grad_output_cache is None:
            self.grad_output_cache = self.setup_cache(output)
            self.grad_cache_iter = 0
        self.update_input_cache(input)
        return output

    def forward_lp(self, input):
        # Need to test do_offset mode whether gradient is updated properly
        input_lp = self.input_cache[self.cache_iter:(
            self.cache_iter + input.size(0))].cuda()
        grad_output_lp = \
            self.grad_output_cache[self.grad_cache_iter:(self.grad_cache_iter + input.size(0))].cuda()
        input_delta = input
        weight_lp = self.weight_lp
        weight_delta = self.weight_delta
        bias_lp = self.bias_lp
        bias_delta = self.bias_delta

        # print("lp forward input ", torch.sum((input_delta + input_lp)**2))


        output = self.lp_func(input_delta, input_lp, grad_output_lp,
                              weight_delta, weight_lp, bias_delta, bias_lp)
        self.cache_iter = (
            self.cache_iter + input.size(0)) % self.n_train_sample
        self.grad_cache_iter = (
            self.grad_cache_iter + input.size(0)) % self.n_train_sample
        return output

    def forward(self, input):
        # Need to test do_offset mode whether gradient is updated properly
        if self.do_offset:
            return self.forward_fp(input)
        else:
            return self.forward_lp(input)
