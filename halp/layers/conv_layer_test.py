import torch
import numpy as np
from torch.nn import Parameter
from halp.layers.conv_layer import BitCenterConv2D, bit_center_conv2d
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from unittest import TestCase
from halp.utils.utils import set_seed
from halp.utils.test_utils import HalpTest
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors, make_jacobian
from halp.layers.bit_center_layer_test import TestBitCenterLayer
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class TestBitCenterConv2DLayer(TestBitCenterLayer, TestCase):
    '''
    Test the functionality of bit centering conv2d layers
    '''

    def get_config(self, type="grad_check"):
        config = {}
        config["input_w"] = 4
        config["input_h"] = 5
        config["kernel_size"] = (3, 3)
        config["stride"] = 1
        config["padding"] = 0
        if type == "grad_check":
            config["n_train_sample"] = 35
            config["dim_in"] = 8
            config["dim_out"] = 16
            config["bias"] = True
            config["cast_func"] = void_cast_func
            config["do_double"] = True
            config["seed"] = 0
            config["batch_size"] = 35
        elif type == "fw_bw_proc":
            config["n_train_sample"] = 98
            config["dim_in"] = 13
            config["dim_out"] = 31
            config["bias"] = True
            config["cast_func"] = single_to_half_det
            config["do_double"] = False
            config["seed"] = 0
            config["batch_size"] = 33
        else:
            raise Exception("Config type not supported!")
        return config

    def prepare_layer(self,
                      input_w,
                      input_h,
                      kernel_size,
                      stride,
                      padding,
                      n_train_sample,
                      dim_in,
                      dim_out,
                      bias,
                      cast_func=void_cast_func,
                      do_double=True,
                      seed=0,
                      batch_size=1):
        layer = BitCenterConv2D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
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
                    dim_out,
                    dim_in,
                    *layer.kernel_size,
                    dtype=torch.double,
                    requires_grad=False).cuda())
            layer.weight_lp.data.copy_(layer.weight.data)
            layer.weight_delta.data.copy_(
                torch.randn(
                    dim_out,
                    dim_in,
                    *layer.kernel_size,
                    dtype=torch.double,
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
                  input_w,
                  input_h,
                  kernel_size,
                  stride,
                  padding,
                  n_train_sample,
                  dim_in,
                  dim_out,
                  bias,
                  cast_func=void_cast_func,
                  do_double=True,
                  seed=0,
                  batch_size=1):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.target_dtype = None

        if do_double:
            input_delta = Parameter(
                torch.randn(
                    n_train_sample,
                    dim_in,
                    input_w,
                    input_h,
                    dtype=torch.double).cuda(),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(
                    n_train_sample,
                    dim_in,
                    input_w,
                    input_h,
                    dtype=torch.double).cuda(),
                requires_grad=True)
        else:
            input_delta = Parameter(
                cast_func(
                    torch.randn(
                        n_train_sample,
                        dim_in,
                        input_w,
                        input_h,
                        dtype=torch.double).cuda()),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(
                    n_train_sample,
                    dim_in,
                    input_w,
                    input_h,
                    dtype=torch.float).cuda(),
                requires_grad=True)
        return [
            input_fp,
        ], [
            input_delta,
        ]


if __name__ == "__main__":
    print(torch.__version__)
    unittest.main()