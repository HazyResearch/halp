import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class BitCenterLinearFunction(Function):
    @staticmethod
    def forward(ctx, input_delta, input_lp, output_grad_lp, 
        weight_delta, weight_lp, bias_delta=None, bias_lp=None):
        # suffix lp means the lp version of the offset tensors
        # suffix delta means the real low precision part of the model representation
        # output_grad_lp is only for backward function, but we need to keep it in ctx
        # for backward function to access it.
        ctx.save_for_backward(input_lp, input_delta, output_grad_lp, 
            weight_lp, weight_delta, bias_lp, bias_delta)
        output = torch.mm(input_delta, weight_lp.t()) \
            + torch.mm((input_lp + input_delta), weight_delta.t())
        if (bias_lp is not None) and (bias_delta is not None):
            # we assume the idx in first dimension represents the sample id in minibatch             
            output += bias_delta.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_lp, input_delta, output_grad_lp, \
            weight_lp, weight_delta, bias_lp, bias_delta = ctx.saved_tensors
        grad_input_lp = None
        grad_input_delta = \
            grad_output.mm((weight_lp + weight_delta)) + output_grad_lp.mm(weight_delta)
        grad_output_grad_lp = None # this dummy to adapt to pytorch API
        grad_weight_lp = None
        grad_weight_delta = \
            torch.mm(grad_output.t(), (input_lp + input_delta)) \
            + output_grad_lp.t().mm(input_delta)
        grad_bias_lp = None
        if (bias_lp is not None) and (bias_delta is not None):
            grad_bias_delta = grad_output.sum(0)
        else:
            grad_bias_delta = None
        return grad_input_delta, grad_input_lp, grad_output_grad_lp, \
            grad_weight_delta, grad_weight_lp, grad_bias_delta, grad_bias_lp


bit_center_linear = BitCenterLinearFunction.apply


class BitCenterLinear(nn.Linear):
    # TODO: consider to sanity check with 32 bit delta terms
    def __init__(self, in_features, out_features, bias=True, 
        cast_func=void_cast_func, n_train_sample=1):
        super(BitCenterLinear, self).__init__(in_features, out_features, bias)
        self.cast_func = cast_func
        # weight_delta is the delta tensor in the algorithm while weight_lp is the cached 
        # lp version of weight offset
        self.weight_delta = Parameter(self.cast_func(self.weight.data), requires_grad=True)
        self.weight_lp = Parameter(self.cast_func(self.weight.data), requires_grad=False)
        self.do_offset = False
        if bias:
            self.bias_delta = Parameter(self.cast_func(self.bias), requires_grad=True)
            self.bias_lp = Parameter(self.cast_func(self.bias.data), requires_grad=False)
        else:
            self.register_parameter('bias_delta', None)
            self.register_parameter('bias_lp', None)
        self.cuda()
        self.reset_parameters_bit_center()
        # input cache
        self.input_cache = None
        self.grad_output_cache = None
        self.cache_iter = 0
        self.grad_cache_iter = 0
        self.n_train_sample = n_train_sample
        # starting from fp mode
        self.set_mode(do_offset=True)
        self.register_backward_hook(self.update_grad_output_cache)

    def reset_parameters_bit_center(self):
        init.zeros_(self.weight_delta)
        if self.bias is not None:
            init.zeros_(self.bias_delta)

    def set_mode(self, do_offset, cache_iter=0):
        self.do_offset = do_offset
        self.cache_iter = cache_iter

    def setup_cache(self, input):
        # the cache is set up when the first minibatch forward is done.
        # here we assume the first dimension of input blob indicates the size of minibatch
        cache_shape = list(input.size())
        cache_shape[0] = self.n_train_sample
        cache = self.cast_func(Variable(torch.zeros(cache_shape).type(input.dtype))).cpu()
        return cache

    def update_grad_output_cache(self, self1, input, output):
        # use duplicated self to adapt to the pytorch API requirement
        # as this is a class member function
        # logger.info(self.__class__.__name__ + " updating grad output cache")
        if self.do_offset:
            self.grad_output_cache[self.grad_cache_iter:min(self.grad_cache_iter + output[0].size()[0], self.n_train_sample)].data.copy_(self.cast_func(output[0].cpu()))
            self.grad_cache_iter = (self.grad_cache_iter + output[0].size(0)) % self.n_train_sample
            # we use the following variable only for test purpose, we want to be able to access
            # the gradeint value wrt input in the outside world. For lp mode, it is grad_input_delta
            # for fp mode, it is grad_input
            self.input_grad_for_test = input[1] # hard code the order of tensor in the input list (can be layer specific)
        else:
            self.input_grad_for_test = input[0]

    def update_input_cache(self, input):
        if self.do_offset:
            self.input_cache[self.cache_iter:min(self.cache_iter + input.size()[0], self.n_train_sample)].data.copy_(self.cast_func(input.cpu()))
            self.cache_iter = (self.cache_iter + input.size(0)) % self.n_train_sample

    def forward_fp(self, input):
        if self.input_cache is None:
            self.input_cache = self.setup_cache(input)
            self.cache_iter = 0
        output = F.linear(input, self.weight, self.bias)
        if self.grad_output_cache is None:
            self.grad_output_cache = self.setup_cache(output)
            self.grad_cache_iter = 0
        self.update_input_cache(input)
        return output

    def forward_lp(self, input):
        # Need to test do_offset mode whether gradient is updated properly
        input_lp = self.input_cache[self.cache_iter:(self.cache_iter + input.size(0))].cuda()
        grad_output_lp = \
            self.grad_output_cache[self.grad_cache_iter:(self.grad_cache_iter + input.size(0))].cuda()
        input_delta = input
        weight_lp = self.weight_lp
        weight_delta = self.weight_delta
        bias_lp = self.bias_lp
        bias_delta = self.bias_delta
        output = bit_center_linear(input_delta, input_lp, grad_output_lp,
            weight_delta, weight_lp, bias_delta, bias_lp)
        self.cache_iter = (self.cache_iter + input.size(0)) % self.n_train_sample
        self.grad_cache_iter = (self.grad_cache_iter + input.size(0)) % self.n_train_sample
        return output

    def forward(self, input):
        # Need to test do_offset mode whether gradient is updated properly
        if self.do_offset:
            return self.forward_fp(input)
        else:
            return self.forward_lp(input)

