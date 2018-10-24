import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from halp.layers.linear_layer import BitCenterLinear, bit_center_linear
from torch.autograd import gradcheck
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from unittest import TestCase
from halp.utils.test_utils import HalpTest
import torch.nn.functional as F
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors, make_jacobian
from itertools import product
import six
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class TestBitCenterLinearLayer(HalpTest, TestCase):
    '''
    Test the functionality of bit centering linear layers
    '''
    @staticmethod
    def test_BitCenterLinearFuncGradientCheck():
        # test if backward is synced with forward in double mode
        # test on a single batch of data, check if the 
        # gradient can give similar numerical loss changes
        # In this test, we use the quadratic loss, this is because
        # using this loss, the grad_output decompose directly to offset and delta
        # in the fp and lp steps
        minibatch_size = 35
        dim_in = 24
        dim_out = 17
        perturb_eps = 1e-6
        rtol_num_analytical_grad = 5e-4
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        for bias in [True, False]:
            for i in range(10):
                layer = BitCenterLinear(dim_in, dim_out, bias=bias, 
                    cast_func=void_cast_func, n_train_sample=minibatch_size)
                layer.double()
                input_delta = torch.randn(minibatch_size, dim_in, dtype=torch.double, requires_grad=True).cuda()
                input_fp = torch.randn(minibatch_size, dim_in, dtype=torch.double, requires_grad=True).cuda()
                output_fp = layer(input_fp) # setup cache using the first forward pass
                layer.weight.data.copy_(torch.randn(dim_out, dim_in, dtype=torch.double, requires_grad=False).cuda())
                layer.weight_lp.data.copy_(layer.weight.data)
                layer.weight_delta.data.copy_(torch.randn(dim_out, dim_in, dtype=torch.double, requires_grad=True).cuda())
                if bias:
                    layer.bias.data.copy_(torch.randn(dim_out, dtype=torch.double, requires_grad=True).cuda())
                    layer.bias_lp.data.copy_(layer.bias.data)
                    layer.bias_delta.data.copy_(torch.randn(dim_out, dtype=torch.double, requires_grad=True).cuda())

                layer.set_mode(do_offset=True)
                output_fp = layer(input_fp)
                output_fp_copy = output_fp.data.clone()
                loss_fp = torch.sum(0.5*output_fp*output_fp)
                loss_fp.backward()
                grad_input_fp = layer.input_grad_for_test.clone()

                layer.set_mode(do_offset=False)
                output_lp = layer(input_delta)
                loss_lp = torch.sum(0.5*output_lp*output_lp)
                loss_lp.backward()
                grad_input_delta = layer.input_grad_for_test.clone()
                weight_grad = layer.weight.grad + layer.weight_delta.grad
                if bias:
                    bias_grad = layer.bias.grad + layer.bias_delta.grad
                # as we only have 1 minibatch, we can directly use layer.grad_output_cache
                input_grad = grad_input_fp + grad_input_delta

                # get numerical finite difference
                layer.set_mode(do_offset=True)
                def get_loss(x):
                    output = layer(x)
                    return torch.sum(0.5*output*output)

                layer.set_mode(do_offset=True)
                layer.weight.data.add_(layer.weight_delta.data)
                if bias:
                    layer.bias.data.add_(layer.bias_delta.data)
                output_final = layer(input_fp + input_delta)
                num_input_grad = get_numerical_jacobian(get_loss, input_fp + input_delta, 
                    target=None, eps=perturb_eps)
                num_weight_grad = get_numerical_jacobian(get_loss, input_fp + input_delta, 
                    target=layer.weight, eps=perturb_eps)
                if bias:
                    num_bias_grad = get_numerical_jacobian(get_loss, input_fp + input_delta, 
                        target=layer.bias, eps=perturb_eps)
                # err = num_input_grad.cpu().numpy().ravel() - input_grad.cpu().numpy().ravel()
                # idx = np.argmax(np.abs(err/input_grad.cpu().numpy().ravel()))
                # print(num_input_grad.cpu().numpy().ravel()[idx], input_grad.cpu().numpy().ravel()[idx], 
                # np.max(np.abs(err/input_grad.cpu().numpy().ravel())))
                # assert forward behavior
                np.testing.assert_allclose(output_final.data.cpu().numpy().ravel(), 
                    (output_fp + output_lp).data.cpu().numpy().ravel(), 
                    rtol=rtol_num_analytical_grad)
                # assert backward behavior
                np.testing.assert_allclose(num_input_grad.cpu().numpy().ravel(), 
                    input_grad.cpu().numpy().ravel(), 
                    rtol=rtol_num_analytical_grad)
                np.testing.assert_allclose(num_weight_grad.cpu().numpy().ravel(), 
                    weight_grad.cpu().numpy().ravel(), 
                    rtol=rtol_num_analytical_grad)
                if bias:
                    np.testing.assert_allclose(num_bias_grad.cpu().numpy().ravel(), 
                        bias_grad.cpu().numpy().ravel(), 
                        rtol=rtol_num_analytical_grad)
        logger.info("Bit centering linear function test passed!")

    def CheckBitCenterLinearBaseTensorProperty(self, layer):
        t_list = [(layer.weight, torch.float32, True, True), 
                  (layer.bias, torch.float32, True, True),
                  (layer.weight_delta, torch.half, True, True),
                  (layer.bias_delta, torch.half, True, True),
                  (layer.weight_lp, torch.half, True, False),
                  (layer.bias_lp, torch.half, True, False),
                  (layer.input_cache, torch.half, False, False)]
        self.CheckLayerTensorProperty(t_list)
        self.CheckLayerTensorGradProperty(t_list)

    # def test_BitCenterLinearLayerFwBwCheck(self, cast_func=single_to_half_det):
    #     # check if the behavior of BitCentering linear layer is going as expected for forward
    #     # the backward behavior is guaranteed by 
    #     # bit center linear function test TestBitCenterLinearFuncGradientCheck
    #     n_sample = 98
    #     n_dim = 13
    #     n_out_dim = 31
    #     minibatch_size = 33
    #     n_minibatch = int(np.ceil(n_sample / minibatch_size))
    #     use_bias = True
    #     layer = BitCenterLinear(in_features=n_dim, out_features=n_out_dim, 
    #         bias=use_bias, cast_func=cast_func, n_train_sample=n_sample)
    #     layer.set_mode(do_offset=True)
    #     layer.cuda()
    #     input_tensor_list = []
    #     if (cast_func == single_to_half_det) or (cast_func == single_to_half_stoc): 
    #         self.CheckBitCenterLinearBaseTensorProperty(layer)
    #     for i in range(n_minibatch):
    #         start_idx = i * minibatch_size
    #         end_idx = min((i + 1) * minibatch_size, n_sample)
    #         if i !=0:
    #             input_cache_before = layer.input_cache[start_idx:end_idx].clone().numpy()
    #             grad_input_cache_before = layer.grad_output_cache[start_idx:end_idx].clone().numpy()
    #         input_tensor = torch.randn(end_idx - start_idx, n_dim, dtype=torch.float32, requires_grad=True).cuda()        
    #         output = layer(input_tensor)
    #         input_tensor_list.append(input_tensor)
    #         torch.sum(output).backward()
    #         if i != 0:
    #             input_cache_after = layer.input_cache[start_idx:end_idx, :].numpy()
    #             grad_input_cache_after = layer.grad_output_cache[start_idx:end_idx, :].numpy()
    #             assert (input_cache_before == 0).all()
    #             assert (grad_input_cache_before == 0).all()
    #             assert (input_cache_before != input_cache_after).all()
    #             assert (grad_input_cache_before != grad_input_cache_after).all()

    #     layer.set_mode(do_offset=False)
    #     if (cast_func == single_to_half_det) or (cast_func == single_to_half_stoc): 
    #         self.CheckBitCenterLinearBaseTensorProperty(layer)
    #     for i in range(n_minibatch):
    #         input_lp = layer.input_cache[layer.cache_iter:(layer.cache_iter + input_tensor_list[i].size(0))].cuda()
    #         output = layer.forward(cast_func(input_tensor_list[i]))
    #         torch.sum(output).backward()
    #         output_ref = torch.mm(cast_func(input_tensor_list[i]), layer.weight_lp.t())\
    #             + torch.mm((input_lp + cast_func(input_tensor_list[i])), layer.weight_delta.t())    
    #         # print("inter type ", layer.weight_lp.t().data.cpu().numpy().dtype)    
    #         # output_ref = np.dot(input_tensor_list[i].data.cpu().numpy().astype(np.float16), layer.weight_lp.t().data.cpu().numpy().astype(np.float16))\
    #         #     + np.dot((input_lp.data.cpu().numpy().astype(np.float16) + input_tensor_list[i].data.cpu().numpy().astype(np.float16)), layer.weight_delta.t().data.cpu().numpy().astype(np.float16))
    #         if use_bias:
    #             output_ref += (layer.bias_delta + layer.bias_lp).unsqueeze(0).expand_as(output_ref)
    #             # output_ref += (layer.bias_delta.data.cpu().numpy().astype(np.float16) + layer.bias_lp.data.cpu().numpy().astype(np.float16))
    #         output_np = output.cpu().data.numpy()
    #         # output_ref_np = output_ref
    #         output_ref_np = output_ref.cpu().data.numpy()
    #         np.testing.assert_array_almost_equal(output_np, output_ref_np)
    #     if (cast_func == single_to_half_det) or (cast_func == single_to_half_stoc): 
    #         self.CheckBitCenterLinearBaseTensorProperty(layer)
    #     print("Bit centering linear layer test passed!")


if __name__ == "__main__":
    print(torch.__version__)
    unittest.mian()
    # test = TestBitCenterLinearLayer()
    # test.test_BitCenterLinearFuncGradientCheck()


