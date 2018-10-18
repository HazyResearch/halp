import torch
import torch.nn as nn
import numpy as np
from halp.utils.test_utils import HalpTest
from halp.optim.bit_center_sgd import BitCenterSGD
from halp.optim.bit_center_svrg import BitCenterSVRG
from unittest import TestCase
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc


# TODO:
# test1, the corresponding cache has the right property, and are property setup for necessary parameters only
# test2, cache are updated and used in the right way
# test3, for step fp check if the gradients is updated properly focus similaly to test 4
# test4, for step lp gradient is updated properly, i.e. focus on the 0 gradients for uninvolved parameters
class TestBitCenterOptim(HalpTest):
    @staticmethod
    def FuncTestCacheProperty(model, optimizer):
        pass

    def test_CacheProperty(self):
        n_train_sample = np.random.randint(low=100, high=1000)
        n_minibatch_per_epoch = np.random.randint(low=10, high=n_train_sample//10)
        model = self.GetMultipleLayerLinearModel(n_layer=3, n_train_sample=n_train_sample)
        optimizer = self.GetOptimizer(model, lr=0.5, weight_decay=0, 
            n_train_sample=n_train_sample, n_minibatch_per_epoch=n_minibatch_per_epoch)
        for param_group, cache_group in zip(optimizer.param_groups, optimizer.grad_cache_groups):
            for p, p_name, cache in zip(param_group["params"], param_group["params_name"], cache_group["cache"]):
                self.FuncTestCacheProperty(p, p_name, cache, optimizer)
        print(self.__class__, " cache property test passed!")

    @staticmethod
    def FuncTestCacheUpdate():
        pass

    @staticmethod
    def TestCacheUpdate():
        pass

    @staticmethod
    def TestStepFP():
        pass

    @staticmethod
    def TestStepLP():
        pass



class TestBitCenterSGD(TestBitCenterOptim, TestCase):
    @staticmethod
    def GetOptimizer(model, lr, weight_decay=0.0, n_train_sample=128,
        cast_func=single_to_half_det, n_minibatch_per_epoch=128):
        params = [x[1] for x in model.named_parameters()]
        names = [x[0] for x in model.named_parameters()]
        return BitCenterSGD(params, names, lr, 
            n_train_sample=n_train_sample, cast_func=cast_func,
            n_minibatch_per_epoch=n_minibatch_per_epoch)

    def FuncTestCacheProperty(self, param, name, cache, optimizer): 
        if cache is None:
            assert not param.requires_grad
        else:
            assert list(param.shape) == list(cache.shape[1:])
            assert cache.size(0) == optimizer.n_minibatch_per_epoch
            assert not name.endswith("_lp")
            t_list = [(cache, torch.float16, False, False)]
            self.CheckLayerTensorProperty(t_list)


class TestBitCenterSVRG(TestBitCenterOptim, TestCase):
    @staticmethod
    def GetOptimizer(model, lr, weight_decay=0.0, n_train_sample=128,
        cast_func=single_to_half_det, n_minibatch_per_epoch=128):
        params = [x[1] for x in model.named_parameters()]
        names = [x[0] for x in model.named_parameters()]
        return BitCenterSVRG(params, names, lr,
            n_train_sample=n_train_sample, cast_func=cast_func,
            n_minibatch_per_epoch=n_minibatch_per_epoch)

    def FuncTestCacheProperty(self, param, name, cache, optimizer): 
        if cache is None:
            assert not param.requires_grad
        else:
            assert list(param.shape) == list(cache.shape)
            assert not name.endswith("_lp")
            t_list = [(cache, torch.float16, True, False)]
            self.CheckLayerTensorProperty(t_list)



if __name__ == "__main__":
    unittest.mian()