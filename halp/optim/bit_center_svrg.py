import torch
import numpy as np
import copy, logging
from torch.autograd import Variable
from torch.optim.optimizer import required, Optimizer
from torch.optim import SGD
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from halp.optim.bit_center_sgd import BitCenterOptim
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


class BitCenterSVRG(BitCenterOptim):
    """
    Implementation of bit centering SVRG
    as the gradient cache is small, we cache gradients on GPU
    We accumulate full gradient in full precision and then cast it
    """
    def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
        n_train_sample=1, cast_func=void_cast_func, minibatch_size=128, T=1):
        super(BitCenterSVRG, self).__init__(params, params_name, lr, 
            weight_decay, n_train_sample, cast_func,
            minibatch_size=minibatch_size, T=T)

    def setup_single_grad_cache(self, grad_shape):
        logger.info("setup fp accum for full grad")
        cache_shape = grad_shape
        return torch.Tensor(np.zeros(cache_shape)).cuda()

    def update_single_grad_cache(self, grad, cache):
        cache.add_(grad)

    def get_single_grad_offset(self, cache):
        # we assume the size of the first dimension is the minibatch size
        return cache

    def on_start_fp_steps(self, model):
        self.setup_grad_cache()
        model.set_mode(do_offset=True)

    def on_end_fp_steps(self, model):
        # for cache_group in self.grad_cache_groups:
        for key, cache in self.grad_cache.items():
            if cache is not None:
                cache.div_(self.n_minibatch_per_epoch)
                self.grad_cache[key] = self.cast_func(cache)
        model.set_mode(do_offset=True)


## the following SVRG is NOT TESTED YET.
# class SVRG(BitCenterOptim):
#     """
#     we hack the bit center SVRG to accommodate fp and lp svrg
#     by reusing the gradient caching system
#     """ 
#     def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
#         n_train_sample=1, minibatch_size=128):
#         """
#         we use void cast function to make sure it behaves in the mode
#         of lp or fp svrg instead of bc svrg
#         """
#         super(SVRG, self).__init__(params, params_name, lr=required, weight_decay=0.0, 
#         n_train_sample=1, cast_func=void_cast_func, minibatch_size=128)

#     def set_model_mode(self, model, do_offset=False):
#         """
#         turn off the functionality to change model mode.
#         as the layers are initialized with do_offset = True
#         SVRG optimizer will keep the model operate in this mode forever.
#         """
#         pass

#     def setup_single_full_grad_cache(self, grad_shape):
#         cache_shape = grad_shape
#         return torch.Tensor(np.zeros(cache_shape)).cuda()

#     def setup_single_grad_cache(self, grad_shape):
#         cache_shape = [self.n_minibatch_per_epoch] + grad_shape
#         return torch.Tensor(np.zeros(cache_shape)).cpu()

#     def update_single_full_grad_cache(self, grad, cache):
#         cache.add_(grad)

#     def update_single_grad_cache(self, grad, cache):
#         # the input grad is actually grad * lr in function update_grad_cache
#         cache[self.cache_iter].copy_(grad.cpu())

#     def setup_grad_cache(self):
#         """
#         We need to setup cache for both full gradient and per_minibatch gradient
#         """
#         self.grad_cache = dict()
#         self.full_grad_cache = dict()
#         for group in self.param_groups:
#             for p, p_name in zip(group["params"], group["params_name"]):
#                 if (not p.requires_grad) \
#                   or (p_name.endswith("_lp")) \
#                   or (p_name.endswith("_delta")):
#                     self.grad_cache[p_name] = None
#                     continue
#                 grad_shape = list(p.size())
#                 cache = self.setup_single_grad_cache(grad_shape)
#                 cache_full = self.setup_single_full_grad_cahce(grad_shape)
#                 self.grad_cache[p_name] = cache
#                 self.full_grad_cache[p_name] = cache_full
#                 logger.info(p_name + " cache setup.")
#         self.cache_iter = 0

#     def update_grad_cache(self):
#         # TODO: need to sanity check when whole dataset size is not divided by the minibatch
#         # to make sure the data idx in each minibatch is the same between the fp pass and lp pass
#         for param_group in self.param_groups:
#             for p, p_name in zip(param_group["params"], param_group["params_name"]):
#                 cache = self.grad_cache[p_name]
#                 cache_full_grad = self.full_grad_cache[p_name]
#                 if cache is None:
#                     continue
#                 self.update_single_grad_cache(p.grad * param_group["lr"], cache)
#                 self.update_single_full_grad_cache(p.grad * param_group["lr"], cache_full_grad)

#     def step_full(self):
#         """
#         update full gradient and cache per minibatch gradient
#         """
#         self.step_fp()

#     def step(self):
#         for param_group in self.param_groups:
#             lr = torch.Tensor(np.array(param_group["lr"])).half()
#             for p, p_name in zip(param_group["params"], param_group["params_name"]):
#                 if p_name.endswith("_delta") or p_name.endswith("_lp"):
#                     continue
#                 cache = self.grad_cache[p_name]
#                 grad_offset = self.get_single_grad_offset(cache)
#                 if p.is_cuda:
#                     lr = lr.cuda()
#                 p.data.add_(-lr, p.grad.data)
#                 if not grad_offset.is_cuda:   
#                    p.data.sub_(grad_offset.cuda())
#                 else:
#                    p.data.sub_(grad_offset)
#         self.step_iter = (self.step_iter + 1) % self.n_minibatch_per_epoch

