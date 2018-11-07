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
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class TestBitCenterLayer(HalpTest):
    '''
    Test the functionality of bit centering linear layers
    the helper functions implemented here should directy serve for linear style
    layers such as linear layer conv layer and etc.
    This template class can benefits parametric layers with one weight param
    and one bias param. E.g. convolutional layers, linear layers
    '''

    def prepare_layer(self, minibatch_size, dim_in, dim_out, bias):
        """
        this function should generate the layer to be tested and the fp lp input.
        Need to specify one parameter in this function for different layers.
        This parameter is self.target_dtype: use None for layers not needing
        target for forward; use torch.float for regression style problems;
        use torch.long for regression style problems
        """
        pass

    def get_analytical_grad(self, layer, input_fp, input_delta, target=None):
        layer.set_mode(do_offset=True)
        grad_list = []
        output_fp = layer(*input_fp)
        output_fp_copy = output_fp.data.clone()
        loss_fp = torch.sum(0.5 * output_fp * output_fp)
        loss_fp.backward()
        grad_input_fp = layer.input_grad_for_test.clone()

        layer.set_mode(do_offset=False)
        output_lp = layer(*input_delta)
        loss_lp = torch.sum(0.5 * output_lp * output_lp)
        loss_lp.backward()
        grad_input_delta = layer.input_grad_for_test.clone()
        # as we only have 1 minibatch, we can directly use layer.grad_output_cache
        input_grad = grad_input_fp + grad_input_delta
        grad_list.append(input_grad)

        weight_grad = layer.weight.grad + layer.weight_delta.grad
        grad_list.append(weight_grad)
        if layer.bias is not None:
            bias_grad = layer.bias.grad + layer.bias_delta.grad
            grad_list.append(bias_grad)
        return output_lp + output_fp, grad_list

    def get_numerical_grad(self,
                           layer,
                           input_fp,
                           input_delta,
                           perturb_eps,
                           target=None):
        # get numerical finite difference
        layer.set_mode(do_offset=True)

        def get_loss(x):
            output = layer(*x)
            return torch.sum(0.5 * output * output)

        grad_list = []
        layer.set_mode(do_offset=True)
        layer.weight.data.add_(layer.weight_delta.data)
        if layer.bias is not None:
            layer.bias.data.add_(layer.bias_delta.data)
        output_final = layer(*[x + y for x, y in zip(input_fp, input_delta)])
        input = []
        for i, (x, y) in enumerate(zip(input_fp, input_delta)):
            input.append(x + y)
        num_input_grad = get_numerical_jacobian(
            get_loss, input, target=input[0], eps=perturb_eps)
        grad_list.append(num_input_grad)
        num_weight_grad = get_numerical_jacobian(
            get_loss, input, target=layer.weight, eps=perturb_eps)
        grad_list.append(num_weight_grad)
        if layer.bias is not None:
            num_bias_grad = get_numerical_jacobian(
                get_loss, input, target=layer.bias, eps=perturb_eps)
            grad_list.append(num_bias_grad)
        return output_final, grad_list

    def test_forward_backward_output(self):
        # test if backward is synced with forward in double mode
        # test on a single batch of data, check if the
        # gradient can give similar numerical loss changes
        # In this test, we use the quadratic loss, this is because
        # using this loss, the grad_output decompose directly to offset and delta
        # in the fp and lp steps
        minibatch_size = 35
        dim_in = 17
        dim_out = 24
        perturb_eps = 1e-6
        rtol_num_analytical_grad = 1e-3
        atol_num_analytical_grad = 1e-6
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        for bias in [True, False]:
            for i in range(10):
                layer = self.prepare_layer(
                    minibatch_size, dim_in, dim_out, bias, do_double=True)
                input_fp, input_delta = self.get_input(
                    minibatch_size,
                    dim_in,
                    dim_out,
                    bias,
                    do_double=True,
                    seed=i + 1)
                analytical_output, analytical_grads = \
                    self.get_analytical_grad(layer, input_fp, input_delta)
                numerical_output, numerical_grads = \
                    self.get_numerical_grad(layer, input_fp, input_delta, perturb_eps)
                assert len(analytical_grads) == len(numerical_grads)

                np.testing.assert_allclose(
                    analytical_output.data.cpu().numpy().ravel(),
                    numerical_output.data.cpu().numpy().ravel(),
                    rtol=rtol_num_analytical_grad)
                for ana_grad, num_grad in zip(analytical_grads,
                                              numerical_grads):
                    if (ana_grad is None) and (num_grad is None):
                        continue

                    np.testing.assert_allclose(
                        ana_grad.data.cpu().numpy().ravel(),
                        num_grad.data.cpu().numpy().ravel(),
                        rtol=rtol_num_analytical_grad,
                        atol=atol_num_analytical_grad * np.max(np.abs(ana_grad.data.cpu().numpy().ravel())))


        logger.info(self.__class__.__name__ + " function test passed!")

    def check_layer_param_and_cache(self, layer):
        t_list = [(layer.weight, torch.float32, True, True),
                  (layer.bias, torch.float32, True, True),
                  (layer.weight_delta, torch.half, True, True),
                  (layer.bias_delta, torch.half, True, True),
                  (layer.weight_lp, torch.half, True, False),
                  (layer.bias_lp, torch.half, True, False),
                  (layer.input_cache, torch.half, False, False),
                  (layer.grad_output_cache, torch.half, False, False)]
        self.CheckLayerTensorProperty(t_list)
        self.CheckLayerTensorGradProperty(t_list)

    def test_layer_forward_backward_precedures(self,
                                               cast_func=single_to_half_det):
        # We test the behavior of layers in a multiple epoch (each with multiple minibatch setting).
        # Along with this, we will also test the property of tensors
        # (including param and cache along the way).
        # check if the behavior of BitCentering linear layer is going as expected for forward
        # the backward behavior is guaranteed by
        # bit center linear function test TestBitCenterLinearFuncGradientCheck
        n_sample = 98
        n_dim = 13
        n_out_dim = 31
        minibatch_size = 33
        n_minibatch = int(np.ceil(n_sample / minibatch_size))
        for use_bias in [True, False]:
            # use_bias = True
            layer = self.prepare_layer(
                n_sample,
                n_dim,
                n_out_dim,
                use_bias,
                cast_func=single_to_half_det,
                do_double=False)
            # test fp mode
            layer.set_mode(do_offset=True)
            layer.cuda()
            input_tensor_list = []
            # target_list = []
            if (cast_func == single_to_half_det) or (
                    cast_func == single_to_half_stoc):
                self.check_layer_param_and_cache(layer)
            for i in range(n_minibatch):
                start_idx = i * minibatch_size
                end_idx = min((i + 1) * minibatch_size, n_sample)
                if i != 0:
                    input_cache_before = layer.input_cache[
                        start_idx:end_idx].clone().numpy()
                    grad_input_cache_before = layer.grad_output_cache[
                        start_idx:end_idx].clone().numpy()
                input_fp, _ = self.get_input(
                    end_idx - start_idx,
                    n_dim,
                    n_out_dim,
                    use_bias,
                    do_double=False)

                output = layer(*input_fp)
                input_tensor_list.append(input_fp)
                torch.sum(output).backward()
                if i != 0:
                    input_cache_after = layer.input_cache[start_idx:
                                                          end_idx, :].numpy()
                    grad_input_cache_after = layer.grad_output_cache[
                        start_idx:end_idx, :].numpy()
                    assert (input_cache_before == 0).all()
                    assert (grad_input_cache_before == 0).all()
                    assert (input_cache_before != input_cache_after).all()
                    assert (grad_input_cache_before !=
                            grad_input_cache_after).all()
            # test lp mode
            layer.set_mode(do_offset=False)
            if (cast_func == single_to_half_det) or (
                    cast_func == single_to_half_stoc):
                self.check_layer_param_and_cache(layer)
            for i in range(n_minibatch):
                # the random seed is controlled, so the target labels should be the same
                # as in the fp iterations
                _, input_delta = self.get_input(
                    end_idx - start_idx,
                    n_dim,
                    n_out_dim,
                    use_bias,
                    cast_func=single_to_half_det,
                    do_double=False)
                output = layer(*input_delta)
                torch.sum(output).backward()
            if (cast_func == single_to_half_det) or (
                    cast_func == single_to_half_stoc):
                self.check_layer_param_and_cache(layer)
        logger.info(self.__class__.__name__ + " layer test passed!")
