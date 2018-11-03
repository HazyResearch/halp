import torch
import numpy as np
import copy, logging
from torch.autograd import Variable
from torch.optim.optimizer import required, Optimizer
from torch.optim import SGD
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
from halp.utils.utils import get_recur_attr
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')

# TODO the weight decay needs updates !!!!!!

class BitCenterOptim(SGD):
    """
    The base class for bit center optimizer: bit center SGD, bit center SVRG
    """
    def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
        n_train_sample=128, cast_func=void_cast_func, minibatch_size=128, T=1):
        """
        The base class for bit centering style optimizers
        The bit centering optimizer can be used with calling step_fp for compute offset
        and with calling step_lp for compute delta.
        The non bit centering version of the same update rule can be implemented only using
        step_fp for updates.
        """
        # TODO (Jian) Considering merge step_fp and step_lp into a single step function
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
        self.T = T

    def setup_single_grad_cache(self):
        # we assume the size of the first dimension is the minibatch size
        pass

    def setup_grad_cache(self):
        self.grad_cache = dict()
        for group in self.param_groups:
            for p, p_name in zip(group["params"], group["params_name"]):
                if (not p.requires_grad) \
                  or (p_name.endswith("_lp")) \
                  or (p_name.endswith("_delta")):
                    self.grad_cache[p_name] = None
                    continue
                grad_shape = list(p.size())
                cache = self.setup_single_grad_cache(grad_shape)
                self.grad_cache[p_name] = cache
                logger.info(p_name + " cache setup.")
        self.cache_iter = 0

    def update_single_grad_cache(self, grad, cache):
        pass

    def update_grad_cache(self):
        # TODO: need to sanity check when whole dataset size is not divided by the minibatch
        # to make sure the data idx in each minibatch is the same between the fp pass and lp pass
        for param_group in self.param_groups:
            weight_decay = param_group["weight_decay"]
            for p, p_name in zip(param_group["params"], param_group["params_name"]):
                cache = self.grad_cache[p_name]
                if cache is None:
                    continue
                if weight_decay != 0.0:
                    p.grad.data.add_(weight_decay, p.data)
                self.update_single_grad_cache(p.grad * param_group["lr"], cache)

    def get_single_grad_offset(self, cache, cache_iter=0):
        # cache iter is useful for bit centering SGD to retrieve gradient offset
        pass

    def step_lp(self):
        for param_group in self.param_groups:
            lr = self.cast_func(torch.Tensor(np.array(param_group["lr"])))
            weight_decay = self.cast_func(torch.Tensor(np.array(param_group["weight_decay"])))
            for p, p_name in zip(param_group["params"], param_group["params_name"]):
                if not p_name.endswith("_delta"):
                    continue
                cache = self.grad_cache[p_name.split("_delta")[0]]
                grad_offset = self.get_single_grad_offset(cache)
                if weight_decay.item() != 0.0:
                    # add the weight decay from the weight_lp style variable
                    # (the one casted from the full precision weight)
                    lp_var_found=False
                    for p_lp, p_lp_name in zip(param_group["params"], param_group["params_name"]):
                        if p_name.replace("_delta", "_lp") == p_lp_name:
                            lp_var_found = True
                            p.grad.data.add_(weight_decay.item(), p_lp.data + p.data)
                            # add weight decay from the delta variable
                    if lp_var_found == False:
                        raise Exception("The lp var is not found for weight decay")
                move = lr.item() * p.grad.data 
                if p.is_cuda:   
                   move.add_(grad_offset.cuda())
                else:
                   move.add_(grad_offset)
                p.data.add_(-move)
        self.step_iter = (self.step_iter + 1) % self.n_minibatch_per_epoch

    def step_fp(self):
        # update all the blobs as usual, the uninvolved blobs has 0 gradient so effectively it is not updated
        # during the entire training
        self.update_grad_cache()
        self.cache_iter = (self.cache_iter + 1) % self.n_minibatch_per_epoch


    def update_offset_vars(self):
        for param_group in self.param_groups:
            for p, p_name in zip(param_group["params"], param_group["params_name"]):
                if not p_name.endswith("_delta"):
                    continue
                # search for the
                corr_found = False 
                for param_offset_group in self.param_groups:
                    for p_offset, p_offset_name in zip(param_offset_group["params"], param_offset_group["params_name"]):
                        if p_offset_name == p_name.split("_delta")[0]:
                            # p_offset = p_offset + p.type(p_offset.dtype)
                            p_offset.data.add_(p.data.type(p_offset.dtype))
                            corr_found = True
                if corr_found == False:
                    logger.error("Can not find offset var for ", p_name)
                    raise Exception("Can not find offset var for ", p_name)

    def clear_cache(self):
        for cache in self.grad_cache.values():
            if cache is None:
                continue
            if not cache.is_cuda:
                cache.copy_(self.cast_func(torch.zeros(cache.size())).cpu())
            else:
                cache.zero_()

    def reset_delta_vars(self):
        for param_group in self.param_groups:
            for p, p_name in zip(param_group["params"], param_group["params_name"]):
                if p_name.endswith("_delta"):
                    p.data.zero_()

    # note we set the mode of model using the following
    # helpers. After each specific fp or lp phase,
    # we set model back to do_offset=True as the defaut
    # statues
    def on_start_lp_steps(self, model):
        self.reset_delta_vars()
        model.set_mode(do_offset=False)
        # self.set_model_mode(model, do_offset=False)

    def on_end_lp_steps(self, model):
        self.update_offset_vars()
        model.set_mode(do_offset=True)
        # self.set_model_mode(model, do_offset=True)

    def on_start_fp_steps(self, model):
        self.clear_cache()
        model.set_mode(do_offset=True)
        # self.set_model_mode(model, do_offset=True)
        
    def on_end_fp_steps(self, model):
        # pass
        model.set_mode(do_offset=True)
        # self.set_model_mode(model, do_offset=True)

    def step(self):
        raise Exception("This function is not suppose to be called. Please use step_lp or step_fp")


class BitCenterSGD(BitCenterOptim):
    """
    Implementation of bit centering SGD
    """
    def __init__(self, params, params_name, lr=required, weight_decay=0.0, 
        n_train_sample=128, cast_func=void_cast_func, minibatch_size=128, T=1):
        super(BitCenterSGD, self).__init__(params, params_name, lr, 
            weight_decay, n_train_sample, cast_func, 
            minibatch_size=minibatch_size, T=T)

    def setup_single_grad_cache(self, grad_shape):
        cache_shape = [self.n_minibatch_per_epoch] + grad_shape
        return self.cast_func(torch.Tensor(np.zeros(cache_shape)).cpu()).cpu()

    def update_single_grad_cache(self, grad, cache):
        # the input grad is actually grad * lr in function update_grad_cache
        cache[self.cache_iter].copy_(self.cast_func(grad.cpu()))

    def get_single_grad_offset(self, cache):
        # we assume the size of the first dimension is the minibatch size
        return cache[self.step_iter]
