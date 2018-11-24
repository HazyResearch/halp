import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import BatchNorm2d
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from halp.layers.bit_center_layer import BitCenterLayer
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("batch norm")


def expand_param_as_input(param, input):
    return param.view(1, input.size(1), 1, 1).expand_as(input)


def sum_tensor_as_param(tensor):
    return tensor.sum(-1).sum(-1).sum(0)


def get_bn_grads(grad_output, weight, input, sigma_sq, mu, eps, x_hat):
    d_x_hat = grad_output * expand_param_as_input(weight, input)
    inv_std = torch.tensor([1.0], device=sigma_sq.device, dtype=sigma_sq.dtype)\
        / torch.sqrt(sigma_sq + eps)

    input_center = input - expand_param_as_input(mu, input)

    # print("input center ", input, mu, input_center)
    # print("inside input ", input)

    d_sigma_sq = -torch.tensor(
        [0.5], dtype=sigma_sq.dtype,
        device=sigma_sq.device) * sum_tensor_as_param(
            d_x_hat * input_center * expand_param_as_input(
                inv_std**3, input_center))

    d_mu = sum_tensor_as_param(
        -d_x_hat * expand_param_as_input(inv_std, d_x_hat)
    ) + torch.tensor(
        [-2.0 / input.size(0)], dtype=sigma_sq.dtype, device=sigma_sq.
        device) * d_sigma_sq * sum_tensor_as_param(input_center)

    # print("inside int ", d_sigma_sq, d_mu)
    # print("inside 1.5 ", sum_tensor_as_param(
    #     -d_x_hat * expand_param_as_input(inv_std, d_x_hat)
    # ), torch.tensor(
    #     [-2.0 / input.size(0)], dtype=sigma_sq.dtype, device=sigma_sq.
    #     device) * d_sigma_sq * sum_tensor_as_param(input_center))
    # print("inside2 ", d_x_hat, input_center, inv_std, sigma_sq, mu)



    d_x = d_x_hat * expand_param_as_input(inv_std, d_x_hat) \
        + torch.tensor([2.0 / input.size(0)], dtype=d_x_hat.dtype, device=d_x_hat.device) \
        * expand_param_as_input(d_sigma_sq, input_center) * input_center \
        + torch.tensor([1.0 / input.size(0)], dtype=d_mu.dtype, device=d_mu.device) \
        * expand_param_as_input(d_mu, input)

    # print("inside 3.0", d_x)


    d_weight = sum_tensor_as_param(grad_output * x_hat)
    d_bias = sum_tensor_as_param(grad_output)

    # print("inside 4.0", d_weight, d_bias)


    return d_x, d_weight, d_bias


class BitCenterBatchNormFunction(Function):
    """
    Notations in this class alignes with the ones in Srgey et al.(https://arxiv.org/pdf/1502.03167.pdf).
    """

    @staticmethod
    def forward(ctx, input_delta, input_lp, mu_delta, mu_lp, sigma_sq_delta,
                sigma_sq_lp, output_grad_lp, weight_delta, weight_lp,
                bias_delta, bias_lp, momentum, eps):
        input_full = input_delta + input_lp
        batch_size = input_delta.size(0)
        eps = torch.tensor([eps],
                           dtype=sigma_sq_delta.dtype,
                           device=sigma_sq_delta.device,
                           requires_grad=False)

        print("\ntest inside pre ", weight_delta + weight_lp, bias_lp + bias_delta, mu_lp + mu_delta, sigma_sq_delta + sigma_sq_lp)



        # we assume input is 4d tensor
        batch_mean = input_full.mean(-1).mean(-1).mean(0).view(
            1, input_full.size(1), 1, 1)
        batch_var = (input_full**2).mean(-1).mean(-1).mean(0).view(1, input_full.size(1), 1, 1) \
         - batch_mean * batch_mean
        # Given O + d <--(O + d) (1 - rho) + rho * V where V is the new observed value.
        # the update rule to delta running statistics d is
        # d <-- d (1 - rho) + rho (V - O)
        mu_delta.mul_(
            torch.Tensor([
                1.0 - momentum,
            ]).type(mu_delta.dtype).item())
        mu_delta.add_(
            torch.Tensor([
                momentum,
            ]).type(mu_delta.dtype).item(),
            batch_mean.squeeze() - mu_lp)
        sigma_sq_delta.mul_(
            torch.Tensor([
                1.0 - momentum,
            ]).type(sigma_sq_delta.dtype).item())
        sigma_sq_delta.add_(
            torch.Tensor([
                momentum,
            ]).type(sigma_sq_delta.dtype).item(),
            batch_var.squeeze() - sigma_sq_lp)

        x_hat_lp = \
         (input_lp - expand_param_as_input(mu_lp, input_lp)) \
         / expand_param_as_input(torch.sqrt(sigma_sq_lp + eps), input_lp)
        x_hat_full = \
         (input_full - expand_param_as_input(mu_lp + mu_delta, input_full)) \
         / expand_param_as_input(torch.sqrt(sigma_sq_lp + sigma_sq_delta + eps), input_full)

        y_lp = expand_param_as_input(weight_lp, input_lp) * x_hat_lp \
            + expand_param_as_input(bias_lp, input_lp)
        y_full = expand_param_as_input(weight_lp + weight_delta, input_full) * x_hat_full \
            + expand_param_as_input(bias_lp + bias_delta, input_full)
        ctx.save_for_backward(input_delta, input_lp, x_hat_lp, x_hat_full,
                              mu_delta, mu_lp, sigma_sq_delta, sigma_sq_lp,
                              output_grad_lp, weight_lp, weight_delta, eps)
        
        print("\ndouble inside check out ", bias_lp + bias_delta, y_full)


        return y_full - y_lp

    def backward(ctx, grad_output):
        input_delta, input_lp, x_hat_lp, x_hat_full, \
        mu_delta, mu_lp, sigma_sq_delta, sigma_sq_lp, \
        output_grad_lp, weight_lp, weight_delta, eps = ctx.saved_tensors

        # print("inside 0 ", grad_output, output_grad_lp, mu_lp, sigma_sq_lp)


        d_x_full, d_weight_full, d_bias_full = \
         get_bn_grads(grad_output + output_grad_lp,
           weight_delta + weight_lp,
           input_delta + input_lp,
           sigma_sq_delta + sigma_sq_lp,
           mu_delta + mu_lp,
           eps,
           x_hat_full)

        # print("inside 0.5 ", grad_output, output_grad_lp, mu_lp, sigma_sq_lp, input_lp)


        d_x_lp, d_weight_lp, d_bias_lp = \
         get_bn_grads(output_grad_lp,
           weight_lp,
           input_lp,
           sigma_sq_lp,
           mu_lp,
           eps,
           x_hat_lp)

        print()
        # print("inside 0.6 ", d_x_full - d_x_lp, d_x_full, d_x_lp)
        print("\ntest inside ", weight_delta + weight_lp, mu_lp + mu_delta, sigma_sq_delta + sigma_sq_lp)
        # print("inside grad ", d_x_full, d_weight_full, d_bias_full)
        print("\ninside grad ", d_x_full)


        return d_x_full - d_x_lp, None, None, None, None, None, None, \
            d_weight_full - d_weight_lp, None, d_bias_full - d_bias_lp, None, None, None


bit_center_batch_norm2d = BitCenterBatchNormFunction.apply


class BitCenterBatchNorm2D(BitCenterLayer, BatchNorm2d):
    """
    This is an implementation of batch norm 2d. It currently
    only support batch norm layer with affine transformation
    """

    def __init__(self,
                 num_features,
                 cast_func,
                 eps=1e-05,
                 momentum=0.1,
                 n_train_sample=1):
        BitCenterLayer.__init__(
            self,
            fp_functional=F.batch_norm,
            lp_functional=bit_center_batch_norm2d,
            bias=True,
            cast_func=cast_func,
            n_train_sample=n_train_sample)
        BatchNorm2d.__init__(
            self,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True)

        # set up delta part of affine transform param
        self.setup_bit_center_vars()
        # set up delta part of the running statistics
        self.setup_bit_center_stat()
        self.cuda()
        # initialize bit center delta parameters (the offset part is initialized by the base BatchNorm2D class)
        self.reset_parameters_bit_center()
        # initialize bit center delta running statistics (the offset part is initialized by the base BatchNorm2D class)
        self.reset_stat_bit_center()
        # register backward hook to update grad cache
        self.register_backward_hook(self.update_grad_output_cache)

    def setup_bit_center_stat(self):
        self.running_mean_delta = \
            Parameter(self.cast_func(self.running_mean), requires_grad=True)
        self.running_mean_lp = \
            Parameter(self.cast_func(self.running_mean), requires_grad=True)

        self.running_var_delta = \
            Parameter(self.cast_func(self.running_var), requires_grad=True)
        self.running_var_lp = \
            Parameter(self.cast_func(self.running_var), requires_grad=True)

    def reset_stat_bit_center(self):
        init.zeros_(self.running_mean_delta)
        init.zeros_(self.running_mean_lp)
        init.zeros_(self.running_var_delta)
        init.zeros_(self.running_var_lp)

    def check_or_setup_input_cache(self, input):
        if self.input_cache is None:
            self.input_cache = self.setup_cache(input)
            self.cache_iter = 0

    def check_or_setup_grad_cache(self, output):
        if self.grad_output_cache is None:
            self.grad_output_cache = self.setup_cache(output)
            self.grad_cache_iter = 0

    def get_input_cache_grad_cache(self, input):
        input_lp = self.input_cache[self.cache_iter:(
            self.cache_iter + input.size(0))].cuda()
        grad_output_lp = \
            self.grad_output_cache[self.grad_cache_iter:(self.grad_cache_iter + input.size(0))].cuda()
        return input_lp, grad_output_lp

    def increment_cache_iter(self, input):
        self.cache_iter = (
            self.cache_iter + input.size(0)) % self.n_train_sample
        self.grad_cache_iter = (
            self.grad_cache_iter + input.size(0)) % self.n_train_sample

    def forward_fp(self, input):

        # print("fp input", input)
        # print("fp weight ", self.weight)
        # print("fp bias ", self.bias)
        # print("fp running mean ", self.running_mean)
        # print("fp running var ", self.running_var)

        self.check_or_setup_input_cache(input)
        # as foward fp is used for test or fp steps
        # it should not update the running statistics
        output = self.fp_func(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            momentum=self.momentum,
            eps=self.eps)
        self.check_or_setup_grad_cache(output)
        self.update_input_cache(input)

        # print("fp output ", torch.sum(output))

        return output

    def forward_lp(self, input):

        # print("lp input", input)
        # print("lp weight ", self.weight_delta)
        # print("lp bias ", self.bias_delta)
        # print("lp running mean ", self.running_mean_delta)
        # print("lp running var ", self.running_var_delta)

        input_lp, grad_output_lp = self.get_input_cache_grad_cache(input)
        # note fp func only has training mode
        output = self.lp_func(
            input, input_lp, self.running_mean_delta, self.running_mean_lp,
            self.running_var_delta, self.running_var_lp, grad_output_lp,
            self.weight_delta, self.weight_lp, self.bias_delta, self.bias_lp,
            self.momentum, self.eps)
        self.increment_cache_iter(input)

        # print("lp output ", torch.sum(output))

        return output