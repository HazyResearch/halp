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
logger = logging.getLogger('bit center sgd')


class BitCenterSVRG(BitCenterOptim):
    """
    Implementation of bit centering SVRG
    """
    def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
        n_train_sample=1, cast_func=void_cast_func, minibatch_size=128):
        super(BitCenterSVRG, self).__init__(params, params_name, lr, 
            weight_decay, n_train_sample, cast_func,
            minibatch_size=minibatch_size)

    def setup_single_grad_cache(self, grad_shape):
        cache_shape = grad_shape
        return self.cast_func(torch.Tensor(np.zeros(cache_shape))).cuda()

    def update_single_grad_cache(self, grad, cache):
        cache.add_(self.cast_func(grad))

    def get_single_grad_offset(self, cache):
        # we assume the size of the first dimension is the minibatch size
        return cache        

    def on_end_fp_steps(self, model):
        # for cache_group in self.grad_cache_groups:
        for cache in self.grad_cache.values():
            if cache is not None:
                cache.div_(self.n_minibatch_per_epoch)
        self.set_model_mode(model, do_offset=True)


