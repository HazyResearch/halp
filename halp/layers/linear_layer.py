import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc

class BitCenterLinearFunction(Function):
    @staticmethod
    def forward(ctx, input_lp, input_delta, weight_lp, weight_delta, bias_lp=None, bias_delta=None):
        # suffix lp means the lp version of the offset tensors
        # suffix delta means the real low precision part of the model representation
        ctx.save_for_backward(input_lp, input_delta, weight_lp, weight_delta, bias_lp, bias_delta)
        output = torch.mm(input_delta, weight_lp.t()) \
            + torch.mm((input_lp + input_delta), weight_delta.t())
        if (bias_lp is not None) and (bias_delta is not None):
            # we assume the idx in first dimension represents the sample id in minibatch             
            output += (bias_lp + bias_delta).unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_lp, input_delta, weight_lp, weight_delta, bias_lp, bias_delta = ctx.saved_tensors
        grad_input_lp = None
        grad_input_delta = grad_output.mm((weight_lp + weight_delta))
        grad_weight_lp = None
        grad_weight_delta = torch.mm(grad_output.t(), (input_lp + input_delta))
        grad_bias_lp = None
        if (bias_lp is not None) and (bias_delta is not None):
            grad_bias_delta = grad_output.sum(0).squeeze(0)
        else:
            grad_bias_delta = None
        return grad_input_lp, grad_input_delta, grad_weight_lp, \
            grad_weight_delta, grad_bias_lp, grad_bias_delta


bit_center_linear = BitCenterLinearFunction.apply


class BitCenterLinearBase(nn.Linear):
    # TODO: consider to sanity check with 32 bit delta terms
    def __init__(self, in_features, out_features, bias=True, 
        cast_func=void_cast_func):
        super(BitCenterLinearBase, self).__init__(in_features, out_features, bias)
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
        self.cuda()
        self.reset_parameters_bit_center()
        # input cache
        self.input_cache = None
        self.grad_input_cache = None

    def reset_parameters_bit_center(self):
        init.zeros_(self.weight_delta)
        if self.bias is not None:
            init.zeros_(self.bias_delta)

    def set_mode(self, do_offset):
        self.do_offset = do_offset


class BitCenterLinear(BitCenterLinearBase):
    def __init__(self, in_features, out_features, bias=True, 
        cast_func=single_to_half_det, n_train_sample=1):
        super(BitCenterLinear, self).__init__(in_features, out_features, bias, cast_func)
        self.cache_iter = 0
        self.n_train_sample = n_train_sample

    def set_mode(self, do_offset, cache_iter=0):
        self.do_offset = do_offset
        self.cache_iter = cache_iter

    def setup_cache(self, input):
        # the cache is set up when the first minibatch forward is done.
        # here we assume the first dimension of input blob indicates the size of minibatch
        cache_shape = list(input.size())
        cache_shape[0] = self.n_train_sample
        cache = self.cast_func(Variable(torch.zeros(cache_shape))).cpu()
        return cache
    # TODO: Test whether forward_fp properly generate forward output and backward output
    # Consider how to adapt to LP SGD and LP SVRG mode
    # def forward_fp(self, input):
    #     return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        # Need to test do_offset mode whether gradient is updated properly
        if self.do_offset:
            if self.input_cache is None:
                self.input_cache = self.setup_cache(input)
                self.cache_iter = 0
            self.input_cache[self.cache_iter:min(self.cache_iter + input.size()[0], self.n_train_sample)].data = self.cast_func(input.cpu())
            self.cache_iter += input.size(0)
            return F.linear(input, self.weight, self.bias)
        else:
            input_lp = self.input_cache[self.cache_iter:(self.cache_iter + input.size(0))].cuda()
            input_delta = input
            weight_lp = self.weight_lp
            weight_delta = self.weight_delta
            bias_lp = self.bias_lp
            bias_delta = self.bias_delta
            output = bit_center_linear(input_lp, input_delta, weight_lp, 
                weight_delta, bias_lp, bias_delta)
            self.cache_iter += input.size(0)
            return output


