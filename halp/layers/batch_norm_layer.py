import torch
import torch.nn as nn
from torch.nn import Linear
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
	return param.view(1, input.size(0), 1, 1).expand_as(input)

def sum_tensor_as_param(tensor):
	return tensor.sum(-1).sum(-1).sum(0)

def get_grads(grad_output, weight, input, sigma_sq, mu, eps, x_hat):
	d_x_hat = grad_output * expand_param_as_input(weight, input)
	inv_std = torch.Tensor([1.0]).type(sigma_sq.dtype) / torch.sqrt(sigma_sq + eps)

	input_center = input - expand_param_as_input(mu)

	d_sigma_sq = -torch.Tensor([0.5]).type(sigma_sq.dtype) * sum_tensor_as_param(
	    d_x_hat * input_center * expand_param_as_input(inv_std**3))

	d_mu = sum_tensor_as_param(-d_x_hat * expand_param_as_input(inv_std**3)) + torch.Tensor([-2.0 / input.size(0)]).type(sigma_sq.dtype) * d_sigma_sq * sum_tensor_as_param(input_center)

	d_x = d_x_hat * expand_param_as_input(inv_std, d_x_hat) \
		+ torch.Tensor([2.0 / input.size(0)]).type(sigma_sq.dtype) \
		* expand_param_as_input(d_sigma_sq, input_center) * input_center \
		+ torch.Tensor([1.0 / input.size(0)]).type(mu.dtype) \
		* expand_param_as_input(d_mu, input)

	d_weight = sum_tensor_as_param(grad_output, x_hat)
	d_bias = sum_tensor_as_param(grad_output)
	return d_x, d_weight, d_bias


class BitCenterBatchNormFunction(Function):
    """
	Notations in this class alignes with the ones in Srgey et al.(https://arxiv.org/pdf/1502.03167.pdf).
	"""

    @staticmethod
    def foward(ctx, input_delta, input_lp, mu_delta, mu_lp, sigma_sq_delta,
               sigma_sq_lp, output_grad_lp, weight_delta, weight_lp, bias_delta, bias_lp,
               exp_aver_fac, eps):
        input_full = input_delta + input_lp
        batch_size = input_delta.size(0)
        # we assume input is 4d tensor
        batch_mean = input_full.mean(-1).mean(-1).mean(0).view(
            1, input_full.size(1), 1, 1)
        batch_var = (input_full**2).mean(-1).mean(-1).mean(0).view(1, input_full.size(1), 1, 1) \
         - batch_mean * batch_mean
        # Given O + d <--(O + d) (1 - rho) + rho * V where V is the new observed value.
        # the update rule to delta running statistics d is
        # d <-- d (1 - rho) + rho (V - O)
        mu_delta.mul_(torch.Tensor(1.0 - exp_aver_fac).type(mu_delta.dtype))
        mu_delta.add_(
            torch.Tensor(exp_aver_fac).type(mu_delta.dtype),
            batch_mean.squeeze())
        sigma_sq_delta.mul_(
            torch.Tensor(1.0 - exp_aver_fac).type(sigma_sq_delta.dtype))
        sigma_sq_delta.add_(
            torch.Tensor(exp_aver_fac).type(sigma_sq_delta.dtype),
            batch_var.squeeze())

        X_hat_lp = \
         (input_lp - expand_param_as_input(mu_lp, input_lp)) \
         / expand_param_as_input(torch.sqrt(sigma_sq_lp + eps), input_lp)
        X_hat_full = \
         (input_full - expand_param_as_input(mu_lp + mu_delta, input_full)) \
         / expand_param_as_input(torch.sqrt(sigma_sq_lp + sigma_sq_delta + eps), input_full)

        y_lp = expand_param_as_input(weight_lp, input_lp) * X_hat_lp \
        	+ expand_param_as_input(bias_lp, input_lp)
        y_full = expand_param_as_input(weight_lp + weight_delta, input_full) * X_hat_full \
         	+ expand_param_as_input(bias_lp + bias_delta, input_full)
        ctx.save_for_backward(input_delta, input_lp, x_hat_lp, x_hat_full,
                              mu_delta, mu_lp, sigma_sq_delta, sigma_sq_lp,
                              output_grad_lp, weight_lp, weight_delta, eps)
        return y_full - y_lp

    def backward(ctx, grad_output):
        input_delta, input_lp, x_hat_lp, x_hat_full, \
        mu_delta, mu_lp, sigma_sq_delta, sigma_sq_lp, \
        output_grad_lp, weight_lp, weight_delta, eps = ctx.saved_tensors

		d_x_full, d_weight_full, d_bias_full = \
		 get_bn_grads(grad_output + output_grad_lp,
		   weight_delta + weight_lp,
		   input_delta + input_lp,
		   sigma_sq_delta + sigma_sq_lp,
		   mu_delta + mu_lp,
		   eps,
		   x_hat_full)

		d_x_lp, d_weight_lp, d_bias_lp = \
		 get_bn_grads(output_grad_lp,
		   weight_lp,
		   input_lp,
		   sigma_sq_lp,
		   mu_lp,
		   eps,
		   x_hat_lp)

		return d_x_full - d_x_lp, None, None, None, None, None, None, \
			d_weight_full - d_weight_lp, None, d_bias_full - d_bias_lp, None, None, None

