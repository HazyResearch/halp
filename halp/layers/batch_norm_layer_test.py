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
import copy
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class TestBitCenterBatchNorm2DLayer(TestBitCenterLayer, TestCase):
    '''
    Test the functionality of bit centering conv2d layers
    '''
    def get_config(self, type="grad_check"):
        config = {}
        # this config can test for padding != 0 and stride > 1 cases
        config["input_w"] = 15
        config["input_h"] = 8
        # config["input_w"] = 2
        # config["input_h"] = 1
        config["eps"] = 1e-5
        config["momentum"] = 0.1
        if type == "grad_check":
            # config["n_train_sample"] = 1
            config["n_train_sample"] = 6
            config["num_features"] = 3
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

    def get_numerical_grad(self,
                           layer,
                           input_fp,
                           input_delta,
                           perturb_eps,
                           target=None):
        # as the running stats change in every forward call,
        # the finite difference approach in bit_center_layer_test.py
        # would not work properly. Instead, we use the original batchnorm2d
        # layer, generate grad and compare to the one we got using 
        # bit center layer.
        grad_list = []
        layer.set_mode(do_offset=True)
        param_dict = layer.state_dict()
        init_running_mean = layer.running_mean.clone() + layer.running_mean_delta.clone()
        init_running_var = layer.running_var.clone() + layer.running_var_delta.clone()

        # update the offset variable
        for name, param in layer.named_parameters():
            if name.endswith("_delta"):
                # print("copied name", name)
                p_offset = param_dict[name.split("_delta")[0]]
                p_offset.data.add_(param)

        param_dict = layer.state_dict()
        layer_orig = torch.nn.BatchNorm2d(num_features=layer.num_features, track_running_stats=True).cuda().double()
        for name, param in layer_orig.named_parameters():
            param.data.copy_(param_dict[name])
        layer_orig.running_mean.data.copy_(init_running_mean.data)
        layer_orig.running_var.data.copy_(init_running_var.data)
        # turn off running stat update for this batch to sync with the bc layer
        layer_orig.train() 
        input = []
        for i, (x, y) in enumerate(zip(input_fp, input_delta)):
            input.append(Parameter(x + y, requires_grad=True))

        output_final = layer_orig(*input)
        loss = 0.5 * torch.sum(output_final**2)
        loss.backward()

        # print("triple check 1 ", torch.sum(output_final**2))
        # print("double check ", torch.sum(layer_orig.running_mean**2))
        # print("double check 2 ", torch.sum(layer_orig.running_var**2))
        # print("inside non bc input grad ", torch.sum(input[0].grad.data**2), torch.sum(layer_orig.weight.grad**2), torch.sum(layer_orig.bias.grad**2))

        grad_list.append(input[0].grad.data.clone())
        grad_list.append(layer_orig.weight.grad.data.clone())
        grad_list.append(layer_orig.bias.grad.data.clone())
        grad_list.append(layer_orig.running_mean.clone())
        grad_list.append(layer_orig.running_var.clone())
        return output_final, grad_list


    def get_analytical_grad(self, layer1, input_fp, input_delta, target=None):
        # this function get the analytical grad with respect to parameters and input
        # it calls get_analytical_param_grad to get grad wrt to paramters.
        # the framework in the function is generic to all layers
        layer = copy.deepcopy(layer1)
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

        grad_list += self.get_analytical_param_grad(layer)
        grad_list += [layer.running_mean.clone() + layer.running_mean_delta.clone()]
        grad_list += [layer.running_var.clone() + layer.running_var_delta.clone()]
        
        # print("triple check 2", torch.sum((output_lp + output_fp)**2))
        # print("double check 3 ", len(grad_list), torch.sum(grad_list[-2]**2), torch.sum(grad_list[-1]**2), torch.sum(grad_list[0]**2), torch.sum(grad_list[1]**2), torch.sum(grad_list[2]**2))

        return output_lp + output_fp, grad_list


if __name__ == "__main__":
    print(torch.__version__)
    unittest.main()

    ##########################
    # manual debugging snippet
    ##########################
    # input_lp = np.array([0.0617,0.6213])
    # input_fp = np.array([0.6614,0.2669])
    # input_full = input_fp + input_lp
    # eps = 1e-5

    # batch_mean = np.mean(input_full)
    # batch_var = np.std(input_full)**2 * input_lp.size / float(input_lp.size - 1)

    # # batch_var = np.mean(input_full**2) - batch_mean**2

    # batch_mean_lp = np.mean(input_lp)
    # batch_var_lp = np.std(input_lp)**2

    # running_mean_full = np.array([0.6472]) + np.array([0.2490])
    # running_var_full = np.array([0.3354]) + np.array([0.4564])

    # # running_mean_full = np.array([0.0])
    # # running_var_full = np.array([1.0])

    # # running_mean_lp = np.array([0.6472])
    # # running_var_lp = np.array([0.3354])

    # weight_full = np.array([0.2699]) + np.array([0.2072])
    # bias_full = np.array([0.2704]) + np.array([0.5507])

    # weight_lp = np.array([0.2072])
    # bias_lp = np.array([0.5507])

    # # running_mean_new = 0.9 * running_mean_full + 0.1 * batch_mean
    # # running_var_new = 0.9 * running_var_full + 0.1 * batch_var

    # running_mean_new = 0.9 * running_mean_full + 0.1 * batch_mean
    # running_var_new = 0.9 * running_var_full + 0.1 * batch_var

    # x_hat_full = ((input_fp + input_lp) - batch_mean) / np.sqrt(batch_var + 1e-5)
    # x_hat_lp = ((input_lp + 0.0) - batch_mean_lp) / np.sqrt(batch_var_lp + 1e-5)

    # y_hat_full = x_hat_full * weight_full + bias_full
    # y_hat_lp = x_hat_lp * weight_lp + bias_lp


    # m = 2
    # d_y_full = y_hat_full
    # d_y_lp = y_hat_lp

    # inv_sigma_full = 1.0/np.sqrt(batch_var + eps)
    # inv_sigma_lp = 1.0/np.sqrt(batch_var_lp + eps)

    # d_x_hat_full = d_y_full * weight_full
    # d_x_hat_lp = d_y_lp * weight_lp

    # # print("outside pre ", d_x_hat_lp, inv_sigma_lp, batch_var_lp + eps, np.sqrt(batch_var_lp + eps), batch_var_lp, batch_mean_lp)


    # d_sigma_sq_full = -np.sum(d_x_hat_full * (input_full - batch_mean) * 0.5 * inv_sigma_full**3)
    # d_sigma_sq_lp = -np.sum(d_x_hat_lp * (input_lp - batch_mean_lp) * 0.5 * inv_sigma_lp**3)

    # d_mu_full = -np.sum(d_x_hat_full * inv_sigma_full) + np.sum(d_sigma_sq_full * -2.0 * (input_full - batch_mean)) / m
    # d_mu_lp = -np.sum(d_x_hat_lp * inv_sigma_lp) + np.sum(d_sigma_sq_lp * -2.0 * (input_lp - batch_mean_lp)) / m


    # # print("outside int ", d_sigma_sq_lp, d_mu_lp)


    # dx_full = d_x_hat_full * inv_sigma_full + d_sigma_sq_full * 2 * (input_full - batch_mean) / m + d_mu_full / m
    # dx_lp =  d_x_hat_lp * inv_sigma_lp + d_sigma_sq_lp * 2 * (input_lp - batch_mean_lp) / m + d_mu_lp / m
    
    # d_weight_full = np.sum(d_y_full * x_hat_full)
    # d_weight_lp = np.sum(d_y_lp * x_hat_lp)

    # d_bias_full = np.sum(d_y_full)
    # d_bias_lp = np.sum(d_y_lp)

    # print("test ", y_hat_full, y_hat_lp, dx_full, dx_lp, running_mean_full, running_var_full, running_mean_new, running_var_new)


    # input_full_t = torch.nn.Parameter(torch.Tensor(input_full).view(1, 1, 2, 1).cuda().cpu(), requires_grad=True)
    # layer_orig = torch.nn.BatchNorm2d(1, momentum=0.1).cuda().cpu()
    # layer_orig.weight.data.copy_(torch.tensor(weight_full, dtype=layer_orig.weight.dtype))
    # layer_orig.bias.data.copy_(torch.tensor(bias_full, dtype=layer_orig.bias.dtype))
    # layer_orig.running_mean.data.copy_(torch.tensor(running_mean_full, dtype=layer_orig.bias.dtype))
    # layer_orig.running_var.data.copy_(torch.tensor(running_var_full, dtype=layer_orig.bias.dtype))
        
    # print("pre test stat ", layer_orig.running_mean.item(), layer_orig.running_var.item())

    # output = layer_orig(input_full_t)
    # loss = torch.sum(0.5 * output * output)
    # loss.backward()
    # print("test layer grad ", output.cpu().detach().numpy(), input_full_t.grad.data.cpu().detach().numpy(), layer_orig.running_mean.item(), layer_orig.running_var.item())


