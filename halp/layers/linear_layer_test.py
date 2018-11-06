import torch
import numpy as np
from torch.nn import Parameter
from halp.layers.linear_layer import BitCenterLinear, bit_center_linear
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

    def prepare_layer(self,
                      n_train_sample,
                      dim_in,
                      dim_out,
                      bias=False,
                      cast_func=void_cast_func,
                      do_double=True):
        layer = BitCenterLinear(
            in_features=dim_in,
            out_features=dim_out,
            bias=bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        # Note do_double = setup layer for gradient check, otherwise, it is for checking
        # the tensor properties
        self.target_dtype = None
        if do_double:
            layer.double()
            # input_delta = torch.randn(n_train_sample, dim_in, dtype=torch.double, requires_grad=True).cuda()
            # input_fp = torch.randn(n_train_sample, dim_in, dtype=torch.double, requires_grad=True).cuda()
            layer.weight.data.copy_(
                torch.randn(
                    dim_out, dim_in, dtype=torch.double,
                    requires_grad=False).cuda())
            layer.weight_lp.data.copy_(layer.weight.data)
            layer.weight_delta.data.copy_(
                torch.randn(
                    dim_out, dim_in, dtype=torch.double,
                    requires_grad=True).cuda())
            if bias:
                layer.bias.data.copy_(
                    torch.randn(
                        dim_out, dtype=torch.double,
                        requires_grad=True).cuda())
                layer.bias_lp.data.copy_(layer.bias.data)
                layer.bias_delta.data.copy_(
                    torch.randn(
                        dim_out, dtype=torch.double,
                        requires_grad=True).cuda())
        layer.cuda()
        return layer

    def get_input(self,
                  n_train_sample,
                  dim_in,
                  dim_out,
                  bias,
                  cast_func=void_cast_func,
                  do_double=True,
                  seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.target_dtype = None

        if do_double:
            input_delta = Parameter(
                torch.randn(n_train_sample, dim_in, dtype=torch.double).cuda(),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(n_train_sample, dim_in, dtype=torch.double).cuda(),
                requires_grad=True)
        else:
            input_delta = Parameter(
                cast_func(
                    torch.randn(n_train_sample, dim_in,
                                dtype=torch.double).cuda()),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(n_train_sample, dim_in, dtype=torch.float).cuda(),
                requires_grad=True)
        return [input_fp,], [input_delta,]


if __name__ == "__main__":
    print(torch.__version__)
    unittest.main()