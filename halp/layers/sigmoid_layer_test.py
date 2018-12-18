import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
from halp.layers.sigmoid_layer import BitCenterSigmoid, bit_center_sigmoid
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from unittest import TestCase
from halp.layers.bit_center_layer_test import TestBitCenterDifferentiableActivationLayer


class TestBitCenterSigmoidLayer(TestBitCenterDifferentiableActivationLayer, TestCase):
    def prepare_layer(self,
                      channel_in,
                      w_in,
                      h_in,
                      cast_func=void_cast_func,
                      bias=False,
                      do_double=True,
                      seed=0,
                      batch_size=1,
                      n_train_sample=1):
        layer = BitCenterSigmoid(
            cast_func=cast_func, n_train_sample=n_train_sample)

        # Note do_double = setup layer for gradient check, otherwise, it is for checking
        # the tensor properties, and layer behaviors
        if do_double:
            layer.double()
        layer.cuda()
        return layer


if __name__ == "__main__":
    print(torch.__version__)
    unittest.main()