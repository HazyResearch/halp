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


# first hard code the input dimensionality 
# and the input output channels
# TODO switch to using config generation function to do generate
# these parameters.
CHANNEL_IN=3
CHANNEL_OUT=5
INPUT_W=17
INPUT_H=17
KERNEL_SIZE=(3, 3)
STRIDE=2
PADDING=0


class TestBitCenterConv2DLayer(TestBitCenterLayer, TestCase):
    '''
    Test the functionality of bit centering linear layers
    '''
    def prepare_layer(self,
                  n_train_sample,
                  dim_in,
                  dim_out,
                  bias,
                  cast_func=void_cast_func,
                  do_double=True):
        layer = BitCenterConv2D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING,
            dilation=1,
            groups=1,
            bias=bias,
            cast_func=cast_func,
            n_train_sample=n_train_sample)

        # print("ckpt 1 ", layer.weight.shape, layer.bias.shape, \
        #     torch.randn(
        #             dim_out, dim_in, *layer.kernel_size, dtype=torch.double,
        #             requires_grad=False).cuda().shape)


        # Note do_double = setup layer for gradient check, otherwise, it is for checking
        # the tensor properties
        self.target_dtype = None
        if do_double:
            layer.double()
            # input_delta = torch.randn(n_train_sample, dim_in, dtype=torch.double, requires_grad=True).cuda()
            # input_fp = torch.randn(n_train_sample, dim_in, dtype=torch.double, requires_grad=True).cuda()
            layer.weight.data.copy_(
                torch.randn(
                    dim_out, dim_in, *layer.kernel_size, dtype=torch.double,
                    requires_grad=False).cuda())
            layer.weight_lp.data.copy_(layer.weight.data)
            layer.weight_delta.data.copy_(
                torch.randn(
                    dim_out, dim_in, *layer.kernel_size, dtype=torch.double,
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
                torch.randn(n_train_sample, dim_in, INPUT_W, INPUT_H, dtype=torch.double).cuda(),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(n_train_sample, dim_in, INPUT_W, INPUT_H, dtype=torch.double).cuda(),
                requires_grad=True)
        else:
            input_delta = Parameter(
                cast_func(
                    torch.randn(n_train_sample, dim_in, INPUT_W, INPUT_H,
                                dtype=torch.double).cuda()),
                requires_grad=True)
            input_fp = Parameter(
                torch.randn(n_train_sample, dim_in, INPUT_W, INPUT_H, dtype=torch.float).cuda(),
                requires_grad=True)
        return [input_fp,], [input_delta,]



# def init_test():
#     set_seed(0)
#     CHANNEL_IN=5
#     CHANNEL_OUT=6
#     BATCH_SIZE=10
#     INPUT_W=4
#     INPUT_H=4
#     KERNEL_SIZE=(3, 3)
#     # INPUT_W=1
#     # INPUT_H=1
#     # KERNEL_SIZE=(1, 1)
#     STRIDE=1
#     PADDING=0

#     input_val = Parameter(torch.randn(BATCH_SIZE, CHANNEL_IN, INPUT_W, INPUT_H), requires_grad=True)
#     layer = torch.nn.Conv2d(CHANNEL_IN, CHANNEL_OUT, KERNEL_SIZE, bias=False)
#     output = layer(input_val)

#     print("input ", input_val.shape, output.shape)

#     loss = torch.sum(output)
#     loss.backward()

#     output_grad = torch.ones_like(output)
#     # input_grad = torch.nn.functional.conv_transpose2d(output_grad, layer.weight.data)
#     # weight_grad = torch.nn.functional.conv_transpose2d(output_grad, input_val)
#     # print("pre check ", output_grad.shape, input_val.shape, output_grad.transpose(0, 1).shape)

#     # print("pt grad weight ", layer.weight.grad, layer.weight.grad.shape)
#     # print("dc grad weight ", weight_grad[:, :, 1:-1, 1:-1], weight_grad.shape)

#     # using the im2col col2im approaches
#     output_grad_unf = output_grad.permute(0, 2, 3, 1).view(output_grad.size(0), -1, output_grad.size(1))
#     input_unf = torch.nn.functional.unfold(input_val, KERNEL_SIZE)

#     # print("test ", output_grad_unf.shape, input_unf.shape)


#     grad_weight = torch.bmm(input_unf, output_grad_unf).sum(dim=0)
#     grad_weight = grad_weight.view(CHANNEL_IN, *KERNEL_SIZE, CHANNEL_OUT).permute(3, 0, 1, 2)
#     weight = layer.weight.view(CHANNEL_OUT, -1)

#     # print("test shape ", output_grad_unf.shape, weight.shape)

#     grad_input = output_grad_unf.matmul(weight)
#     grad_input = torch.nn.functional.fold(grad_input.transpose(1, 2), 
#         output_size=(INPUT_W, INPUT_H), kernel_size=KERNEL_SIZE)
#     print("pt grad weight ", torch.sum(layer.weight.grad**2), layer.weight.grad.shape)
#     print("dc grad weight ", torch.sum(grad_weight**2), grad_weight.shape)

#     print("pt grad input ", torch.sum(input_val.grad**2), input_val.grad.shape)
#     print("dc grad input ", torch.sum(grad_input**2), grad_input.shape)


if __name__ == "__main__":
    print(torch.__version__)
    unittest.main()

    # init_test()

    # unfold = torch.nn.Unfold(kernel_size=(2, 3))
    # input = torch.randn(2, 5, 3, 4)
    # output = unfold(input)
    # output.size()
    
    # # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
    # inp = torch.randn(1, 3, 10, 12) 
    # w = torch.randn(2, 3, 4, 5)
    # inp_unf = torch.nn.functional.unfold(inp, (4, 5))
    # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    # out_unf[:] = 1.0
    # out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))

    # print("check output ", out)

    # print((torch.nn.functional.conv2d(inp, w) - out).abs().max())
