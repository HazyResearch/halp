import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from halp.layers.linear_layer import BitCenterLinear, bit_center_linear
from torch.autograd import gradcheck
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from unittest import TestCase
from halp.utils.test_utils import HalpTest
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors, make_jacobian
from halp.layers.bit_center_layer_test import TestBitCenterLayer
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class TestBitCenterLinearLayer(TestBitCenterLayer, TestCase):
    '''
    Test the functionality of bit centering linear layers
    '''
    def prepare_layer(self, n_train_sample, dim_in, dim_out, bias, cast_func=void_cast_func, do_double=True):
        layer = BitCenterLinear(in_features=dim_in, out_features=dim_out, 
                    bias=bias, cast_func=cast_func, n_train_sample=n_train_sample)
        # Note do_double = setup layer for gradient check, otherwise, it is for checking 
        # the tensor properties
        if do_double:
            layer.double()
            input_delta = torch.randn(n_train_sample, dim_in, dtype=torch.double, requires_grad=True).cuda()
            input_fp = torch.randn(n_train_sample, dim_in, dtype=torch.double, requires_grad=True).cuda()
            output_fp = layer(input_fp) # setup cache using the first forward pass
            layer.weight.data.copy_(torch.randn(dim_out, dim_in, dtype=torch.double, requires_grad=False).cuda())
            layer.weight_lp.data.copy_(layer.weight.data)
            layer.weight_delta.data.copy_(torch.randn(dim_out, dim_in, dtype=torch.double, requires_grad=True).cuda())
            if bias:
                layer.bias.data.copy_(torch.randn(dim_out, dtype=torch.double, requires_grad=True).cuda())
                layer.bias_lp.data.copy_(layer.bias.data)
                layer.bias_delta.data.copy_(torch.randn(dim_out, dtype=torch.double, requires_grad=True).cuda())
        else:
            input_fp = None
            input_delta = None
        return layer, input_fp, input_delta


if __name__ == "__main__":
    print(torch.__version__)
    unittest.mian()