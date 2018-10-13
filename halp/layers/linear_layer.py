import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import single_to_half_det, single_to_half_stoc


class BitCenterLinear(nn.Linear):
    # TODO: consider to sanity check with 32 bit delta terms
    def __init__(self, in_features, out_features, bias=True, 
        cast_func=single_to_half_det, n_train_sample=1):
        super(BitCenterLinear, self).__init__(in_features, out_features, bias)
        self.cast_func = cast_func
        # TODO check if this is a parameter after casting
        # weight_delta is the delta tensor in the algorithm while weight_lp is the cached 
        # lp version of weight
        # TODO make weight_lp bias_lp all in no-gradient mode
        # TODO check if weight delta is with gradient
        self.weight_delta = Parameter(self.cast_func(self.weight.data), requires_grad=True)
        self.weight_lp = Parameter(self.cast_func(self.weight.data), requires_grad=False)
        self.do_offset = False
        if bias:
            self.bias_delta = Parameter(self.cast_func(self.bias), requires_grad=True)
            self.bias_lp = Parameter(self.cast_func(self.bias.data), requires_grad=False)
        else:
            self.register_parameter('bias_delta', None)
            self.register_parameter('bias_lp', None)
        # pre allocate memory for the weight and bias converted to lp
        # to enable weight bias initialization, we need 
        self.cuda()
        self.reset_parameters_bit_center()
        # input cache
        self.input_cache = None
        self.grad_input_cache = None
        self.cache_iter = None
        self.n_train_sample = n_train_sample

    def reset_parameters_bit_center(self):
        init.zeros_(self.weight_delta)
        if self.bias is not None:
            init.zeros_(self.bias_delta)

    def set_mode(do_offset):
        self.do_offset = offset

    def setup_cache(self, input):
        # the cache is set up when the first minibatch forward is done.
        # here we assume the first dimension of input blob indicates the size of minibatch
        cache_shape = list(input.get_shape())
        cache_shape[0] = self.n_train_sample
        cache = Variable(torch.zeros(cache_shape)).cpu()
        return cache
        
    # TODO: Test whether forward_fp properly generate forward output and backward output
    # Consider how to adapt to LP SGD and LP SVRG mode
    # def forward_fp(self, input):
    #     return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        if self.do_offset:
            if self.input_cache is None:
                self.input_cache = self.setup_cache(input)
            self.input_cache[self.cache_iter:min(self.cache_iter + input.get_shape()[0], self.n_train_sample)] = \
                self.cast_func(input.data)
            return F.linear(input, self.weight, self.bias)
        else:
            # For here input is delta_x
            # print("inside ", self.weight_lp.dtype, input.dtype)
            # self.weight_lp.add_(self.weight_lp)
            # print("test tensor types ", self.weight.is_cuda, self.weight_lp.is_cuda, input.is_cuda)
            # torch.mm(self.weight_lp, input)
            return torch.mm(self.weight_lp, input) + torch.mm(self.weight_delta, (self.input_lp + input))