import torch
import numpy as np
from torch.nn import Parameter
from halp.layers.batch_norm_layer import BitCenterBatchNorm2D, bit_center_batch_norm2d
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


class TestBitCenterBatchNorm2DLayer(TestBitCenterLayer, TestCase):
    '''
    Test the functionality of bit centering conv2d layers
    '''
    def get_config(self, type="grad_check"):
        config = {}
        # this config can test for padding != 0 and stride > 1 cases
        # config["input_w"] = 15
        # config["input_h"] = 8
        config["input_w"] = 2
        config["input_h"] = 1
        config["eps"] = 1e-5
        config["momentum"] = 0.1
        if type == "grad_check":
            config["n_train_sample"] = 1
            # config["n_train_sample"] = 6
            config["num_features"] = 1
            config["bias"] = True   # this is dummy
            config["cast_func"] = void_cast_func
            config["do_double"] = True
            config["seed"] = 0
            config["batch_size"] = 6
        elif type == "fw_bw_proc":
            config["n_train_sample"] = 98
            config["num_features"] = 13
            config["bias"] = True # dummy
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
                      n_train_sample,
                      num_features,
                      bias,
                      cast_func,
                      eps,
                      momentum,
                      do_double,
                      seed,
                      batch_size):
        layer = BitCenterBatchNorm2D(
            num_features=num_features,
            cast_func=cast_func,
            eps=eps,
            momentum=momentum,
            n_train_sample=n_train_sample)

        # Note do_double = setup layer for gradient check, otherwise, it is for checking
        # the tensor properties
        if do_double:
            layer.double()
            # properly setup value for weights
            layer.weight.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=True).cuda())
            layer.weight_lp.data.copy_(layer.weight.data)
            layer.weight_delta.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=True).cuda())
            layer.bias.data.copy_(
                torch.randn(
                    layer.num_features, 
                    dtype=torch.double,
                    requires_grad=True).cuda())
            layer.bias_lp.data.copy_(layer.bias.data)
            layer.bias_delta.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=True).cuda())
            # properly setup running statistics
            layer.running_mean.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=False).cuda())
            layer.running_mean_lp.data.copy_(layer.running_mean.data)
            layer.running_mean_delta.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=False).cuda())
            layer.running_var.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=False).cuda()).abs_()
            layer.running_var_lp.data.copy_(layer.running_var.data)
            layer.running_var_delta.data.copy_(
                torch.randn(
                    layer.num_features,
                    dtype=torch.double,
                    requires_grad=False).cuda()).abs_()

        layer.cuda()
        return layer

    def get_input(self,
                  input_w,
                  input_h,
                  n_train_sample,
                  num_features,
                  bias,
                  cast_func,
                  eps,
                  momentum,
                  do_double,
                  seed,
                  batch_size):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.target_dtype = None

        if do_double:
            input_delta = Parameter(
                torch.randn(
                    n_train_sample,
                    num_features,
                    input_w,
                    input_h,
                    dtype=torch.double).cuda(),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(
                    n_train_sample,
                    num_features,
                    input_w,
                    input_h,
                    dtype=torch.double).cuda(),
                requires_grad=True)
        else:
            input_delta = Parameter(
                cast_func(
                    torch.randn(
                        n_train_sample,
                        num_features,
                        input_w,
                        input_h,
                        dtype=torch.double).cuda()),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(
                    n_train_sample,
                    num_features,
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