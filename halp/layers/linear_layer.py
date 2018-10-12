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
    def __init__(self, in_features, out_features, bias=True, cast_func=single_to_half_det):
        super(BitCenterLinear, self).__init__(in_features, out_features, bias)
        self.cast_func = cast_func
        # TODO check if this is a parameter after casting
        # weight_delta is the delta tensor in the algorithm while weight_lp is the cached 
        # lp version of weight
        # TODO make weight_lp bias_lp all in no-gradient mode
        # TODO check if weight delta is with gradient
        self.weight_delta = self.cast_func(self.weight)
        self.weight_lp = Variable(self.cast_func(self.weight.data), requires_grad=False)
        self.do_offset = False
        if bias:
            self.bias_delta = self.cast_func(self.bias)
            self.bias_lp = Variable(self.cast_func(self.bias.data), requires_grad=False)
        else:
            self.register_parameter('bias_delta', None)
            self.register_parameter('bias_lp', None)
        # pre allocate memory for the weight and bias converted to lp
        self.reset_parameters_bit_center()

    def reset_parameters_bit_center(self):
        # init.zeros_(self.weight_delta)
        # self.weight_delta = self.weight_delta.fill_(0.0)
        self.weight_delta = self.cast_func(torch.zeros_like(self.weight_delta.float()))
        if self.bias is not None:
            self.bias_delta = self.cast_func(torch.zeros_like(self.bias_delta.float()))
            # init.zeros_(self.bias_delta)

    def set_mode(do_offset):
        self.do_offset = offset

    # TODO: Test whether forward_fp properly generate forward output and backward output
    # Consider how to adapt to LP SGD and LP SVRG mode
    # def forward_fp(self, input):
    #     return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        if self.do_offset:
            self.input_lp = Parameter(self.cast_func(input.data), require_grad=False)
            return F.linear(input, self.weight, self.bias)
        else:
            # For here input is delta_x
            # print("inside ", self.weight_lp.dtype, input.dtype)
            # self.weight_lp.add_(self.weight_lp)
            # torch.matmul(self.weight_lp, input)
            return torch.mm(self.weight_lp, input) + torch.mm(self.weight_delta, (self.input_lp + input))