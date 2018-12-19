import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from halp.utils.utils import single_to_half_det, single_to_half_stoc
from halp.utils.utils import copy_layer_weights, copy_module_weights
from halp.utils.utils import void_cast_func, get_recur_attr
from halp.layers.bit_center_layer import BitCenterModule
from halp.layers.linear_layer import BitCenterLinear
from halp.layers.cross_entropy import BitCenterCrossEntropy
from halp.layers.sigmoid_layer import BitCenterSigmoid
from halp.layers.tanh_layer import BitCenterTanh
from halp.layers.embedding import BitCenterEmbedding
from halp.layers.ele_mult import BitCenterEleMult
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')


class BitCenterLSTMCell(BitCenterModule, nn.LSTMCell):
    '''
	Implementation of the LSTM cell
	'''

    def __init__(input_size,
                 hidden_size,
                 bias=True,
                 cast_func=void_cast_func,
                 n_train_sample=1):
        BitCenterModule.__init__()
        # nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.cast_func = cast_func
        self.n_train_sample = n_train_sample

        # we process the input and last hidden state in a batch for the 4 gates
        self.input_linear = BitCenterLinear(
            input_size,
            hidden_size * 4,
            bias=self.bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        self.hidden_linear = BitCenterLinear(
            hidden_size,
            hidden_size * 4,
            bias=self.bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        # for the naming of the symbols like i, f, g, o, please refer to 
        # https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        self.i_activation = BitCenterSigmoid(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.f_activation = BitCenterSigmoid(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.g_activation = BitCenterTanh(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.o_activation = BitCenterSigmoid(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)


        self.f_c_mult = BitCenterEleMult(cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.i_g_mult = BitCenterEleMult(cast_func=self.cast_func, n_train_sample=self.n_train_sample)

        self.c_prime_activation = BitCenterTanh(
            cast_func=self.cast_func, n_train_sample=self.n_train_sample)
        self.o_c_prime_mult = BitCenterEleMult(cast_func=self.cast_func, n_train_sample=self.n_train_sample)


    def forward(self, x, (h, c)):
        trans_input = self.input_linear(x)
        trans_hidden = self.input_linear(h)

        out = trans_input + trans_hidden
        i = self.i_activation(out[0:self.hidden_size])
        f = self.f_activation(out[self.hidden_size:(2 * self.hidden_size)])
        g = self.g_activation(out[(2 * self.hidden_size):(3 * self.hidden_size)])
        o = self.o_activation(out[3 * self.hidden_size:])

        c_prime = self.f_c_mult(f, c) + self.i_g_mult(i, g)
       	c_prime_act = self.c_prime_activation(c_prime)
       	h_prime = self.o_c_prime_mult(o, c_prime_act)

       	return (h_prime, c_prime)