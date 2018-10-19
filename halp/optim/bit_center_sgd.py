import torch
import numpy as np
import copy, logging
from torch.autograd import Variable
from torch.optim.optimizer import required, Optimizer
from torch.optim import SGD
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('bit center sgd')


class BitCenterOptim(SGD):
    """
    The base class for bit center optimizer: bit center SGD, bit center SVRG
    """
    def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
        n_train_sample=128, cast_func=void_cast_func, minibatch_size=128):
        """
        The base class for bit centering style optimizers
        Argument:
        """
        # TODO setup gradient cache using a member function
        # Question: how to deal with the zero gradient thing?
        # Solution 1: remember to clear gradient everytime so that we can add the zero gradient for those not involved parameters
        # Solution 2: distinguish with variable name?
        # TODO Define a fp step function to compute and catch gradient
        # TODO Define a lp step function to compute and update delta
        # The Design principle: 1. bit centering optimizer by calling step_lp and step_fp
        # 2. the full precision base optimizer can be implemented via only calling step_fp
        # TODO Consider add the parameter T to decide where to switch fp and lp step 
        # 
        # TODO double check make sure we when a tensor is not involved, the gradient is 0 if it uses
        # model update operation.
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(BitCenterOptim, self).__init__(params, **defaults)
        if len(self.param_groups) != 1:
            logger.error("Bit centering optimizers doesn't support per-parameter options "
                             "(parameter groups)")
            raise ValueError("Bit centering optimizers doesn't support per-parameter options " \
                             "(parameter groups)")
        self.param_groups[0]['params_name'] = params_name
        self.n_train_sample = n_train_sample
        self.n_minibatch_per_epoch = int(np.floor( (self.n_train_sample - 1) // float(minibatch_size)) + 1)
        # self.n_minibatch_per_epoch = n_minibatch_per_epoch
        self.cast_func = cast_func
        self.step_iter = 0    # this is a iter for step_lp function
        self.cache_iter = 0   # this is a iter for updating the gradient cache
        self.setup_grad_cache()



    def setup_single_grad_cache(self):
        # we assume the size of the first dimension is the minibatch size
        pass

    def setup_grad_cache(self):
        self.grad_cache_groups = []
        for group in self.param_groups:
            cache_group = dict()
            cache_group["cache"] = []
            for p, p_name in zip(group["params"], group["params_name"]):
                if (not p.requires_grad) \
                  or (p_name.endswith("_lp")) \
                  or (p_name.endswith("_delta")):
                    cache_group["cache"].append(None)
                    continue
                grad_shape = list(p.size())
                cache = self.setup_single_grad_cache(grad_shape)
                cache_group["cache"].append(cache)
                logger.info(p_name + " cache setup.")
            self.grad_cache_groups.append(cache_group)
        self.cache_iter = 0

    def update_single_grad_cache(self, grad, cache):
        pass

    def update_grad_cache(self):
        # TODO: need to sanity check when whole dataset size is not divided by the minibatch
        # to make sure the data idx in each minibatch is the same between the fp pass and lp pass
        for param_group, cache_group in zip(self.param_groups, self.grad_cache_groups):
            for p, cache in zip(param_group["params"], cache_group["cache"]):
                if cache is None:
                    continue
                self.update_single_grad_cache(p.grad, cache)
        self.cache_iter = (self.cache_iter + 1) % self.n_minibatch_per_epoch

    def get_single_grad_offset(self, cache, cache_iter=0):
        # cache iter is useful for bit centering SGD to retrieve gradient offset
        pass

    def step_lp(self):
        for param_group, cache_group in zip(self.param_groups, self.grad_cache_groups):
            lr = torch.Tensor(np.array(group["lr"])).half()
            for p, cache in zip(param_group["params"], cache_group["cache"]):
                if not p.requires_grad:
                    if cache is not None:
                        logger.error("Suspicious cache exists for no-grad parameter")
                        raise ValueError("Suspicious cache exists for no-grad parameter")
                    continue
                grad_offset = self.get_single_grad_offset(cache)
                p.data.add_(-lr, p.grad.data)
                p.data.sub_(grad_offset)
        self.step_iter = (self.step_iter + 1) % self.n_minibatch_per_epoch

    def step_fp(self):
        # update all the blobs as usual, the uninvolved blobs has 0 gradient so effectively it is not updated
        # TODO add sanity check and assertion to make sure the parameters comes out in the same order
        # during the entire training
        super(BitCenterOptim, self).step()
        self.update_grad_cache()


    # def step(self):
    #   # avoid exposing the complexity to the outside on the use of different function at different steps
    #   if self.n_iter % :

class BitCenterSGD(BitCenterOptim):
    """
    Implementation of bit centering SGD
    """
    def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
        n_train_sample=128, cast_func=void_cast_func, minibatch_size=128):
        super(BitCenterSGD, self).__init__(params, params_name, lr, 
            weight_decay, n_train_sample, cast_func, 
            minibatch_size=minibatch_size)

    def setup_single_grad_cache(self, grad_shape):
        cache_shape = [self.n_minibatch_per_epoch] + grad_shape
        return self.cast_func(torch.Tensor(np.zeros(cache_shape)).cpu()).cpu()

    def update_single_grad_cache(self, grad, cache):
        cache[self.cache_iter].copy_(self.cast_func(grad.cpu()))

    def get_single_grad_offset(self, cache):
        # we assume the size of the first dimension is the minibatch size
        return cache[self.n_iter]



