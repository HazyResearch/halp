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
        minibatch_size = np.random.randint(low=10, high=n_train_sample//10)
        model = self.GetMultipleLayerLinearModel(n_layer=3, n_train_sample=n_train_sample)
        optimizer = self.GetOptimizer(model, lr=0.5, weight_decay=0, 
            n_train_sample=n_train_sample, minibatch_size=minibatch_size)
        for param_group, cache_group in zip(optimizer.param_groups, optimizer.grad_cache_groups):
            for p, p_name, cache in zip(param_group["params"], param_group["params_name"], cache_group["cache"]):
                self.FuncTestCacheProperty(p, p_name, cache, optimizer)
        print(self.__class__, " cache property test passed!")

    @staticmethod
    def FuncTestCacheUpdate():
        pass

    def test_StepFP(self):
        # test grad cache is udpated properly
        # test the involved grad got generated
        n_train_sample = np.random.randint(low=100, high=1000)
        minibatch_size = np.random.randint(low=9, high=n_train_sample//10)
        model = self.GetMultipleLayerLinearModel(n_layer=3, n_train_sample=n_train_sample)
        optimizer = self.GetOptimizer(model, lr=0.0005, weight_decay=0, 
            n_train_sample=n_train_sample, minibatch_size=minibatch_size)
        n_minibatch = int(np.ceil(n_train_sample / float(minibatch_size)))      
        # test in 3 consecutive epochs
        # step_fp is updating the cache properly
        for k in range(3):
            for i in range(n_minibatch):
                start_idx = i * minibatch_size
                end_idx = min((i + 1) * minibatch_size, n_train_sample)
                fw_input = torch.Tensor(np.random.randn(end_idx - start_idx, model.n_feat_in[0])).cuda()
                fw_label = torch.Tensor(np.random.randn(end_idx - start_idx, 1)).cuda()
                loss = model.forward(fw_input, fw_label)
                loss.backward()
                # get the grad cache before fp step
                if k == 0 and i == 0:
                    optimizer.step_fp()
                else:
                    cache_list_before_update = \
                        self.GetUpdatedCache(minibatch_idx=i, optimizer=optimizer)
                    optimizer.step_fp()
                    # get the grad cache after fp step
                    cache_list_after_update = \
                        self.GetUpdatedCache(minibatch_idx=i, optimizer=optimizer)
                    for cache_before, cache_after in \
                        zip(cache_list_before_update, cache_list_after_update):
                        if (cache_before is None) and (cache_after is None):
                            continue
                        assert not (cache_before.cpu().numpy() == cache_after.cpu().numpy()).all()
            # clear cache to 0 for next round test
            for param_group, cache_group in zip(optimizer.param_groups, optimizer.grad_cache_groups):
                for p, p_name, cache in zip(param_group["params"], param_group["params_name"], cache_group["cache"]):
                    if cache is None:
                        continue
                    if not cache.is_cuda:
                        cache.copy_(optimizer.cast_func(torch.zeros(cache.size())).cpu())
                    else:
                        cache.zero_()

    @staticmethod
    def test_StepLP():
        pass



class TestBitCenterSGD(TestBitCenterOptim, TestCase):
    @staticmethod
    def GetOptimizer(model, lr, weight_decay=0.0, n_train_sample=128,
        cast_func=single_to_half_det, minibatch_size=128):
        params = [x[1] for x in model.named_parameters()]
        names = [x[0] for x in model.named_parameters()]
        return BitCenterSGD(params, names, lr, 
            n_train_sample=n_train_sample, cast_func=cast_func,
            minibatch_size=minibatch_size)

    def FuncTestCacheProperty(self, param, name, cache, optimizer): 
        if (cache is None) :
            assert (not param.requires_grad) \
                or (name.endswith("_lp")) \
                or (name.endswith("_delta"))
        else:
            assert list(param.shape) == list(cache.shape[1:])
            assert cache.size(0) == optimizer.n_minibatch_per_epoch
            assert (not name.endswith("_lp") ) and (not name.endswith("_delta") )
            # assert on GPU and not require_grad
            t_list = [(cache, torch.float16, False, False)]
            self.CheckLayerTensorProperty(t_list)

    @staticmethod
    def GetUpdatedCache(minibatch_idx, optimizer):
        cache_list = []
        for param_group, cache_group in zip(optimizer.param_groups, optimizer.grad_cache_groups):
            for p, p_name, cache in zip(param_group["params"], param_group["params_name"], cache_group["cache"]):
                if cache is not None:
                    # print("check idx ", minibatch_idx, cache.size(), cache.cuda()[minibatch_idx])
                    cache_list.append(cache[minibatch_idx].clone())
        return cache_list



class TestBitCenterSVRG(TestBitCenterOptim, TestCase):
    @staticmethod
    def GetOptimizer(model, lr, weight_decay=0.0, n_train_sample=128,
        cast_func=single_to_half_det, minibatch_size=128):
        params = [x[1] for x in model.named_parameters()]
        names = [x[0] for x in model.named_parameters()]
        return BitCenterSVRG(params, names, lr,
            n_train_sample=n_train_sample, cast_func=cast_func,
            minibatch_size=minibatch_size)

    def FuncTestCacheProperty(self, param, name, cache, optimizer): 
        if cache is None:
            assert (not param.requires_grad) \
                or (name.endswith("_lp")) \
                or (name.endswith("_delta"))
        else:
            assert list(param.shape) == list(cache.shape)
            assert (not name.endswith("_lp") ) and (not name.endswith("_delta") )
            # assert on GPU and not require_grad
            t_list = [(cache, torch.float16, True, False)]
            self.CheckLayerTensorProperty(t_list)

    @staticmethod
    def GetUpdatedCache(minibatch_idx, optimizer):
        cache_list = []
        for param_group, cache_group in zip(optimizer.param_groups, optimizer.grad_cache_groups):
            for p, p_name, cache in zip(param_group["params"], param_group["params_name"], cache_group["cache"]):
                if cache is not None:
                    cache_list.append(cache.clone())
        return cache_list



if __name__ == "__main__":
    unittest.mian()