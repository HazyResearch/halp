import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from halp.layers.linear_layer import BitCenterLinear, bit_center_linear
from torch.autograd import gradcheck
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc
# from halp.models.logistic_regression import LogisticRegression


# def GetToyBitCenterLogistic():
#     n_sample = 4
#     n_dim = 3
#     n_class = 4
#     X = Variable(torch.DoubleTensor(np.random.normal(size=(n_sample, n_dim) ) ) )
#     Y = Variable(torch.LongTensor(np.array([0, 1, 3, 2] ) ) )
#     regressor = LogisticRegression(input_dim=n_dim, n_class=n_class, reg_lambda=100.0, dtype="half")
#     # loss1 = regressor.forward(X, Y)
#     # loss_diff = 0.0
#     # move = 1e-9
#     # loss1.backward()
#     # for w in regressor.parameters():
#     # loss_diff += torch.sum(w.grad.data * move)
#     # for w in regressor.parameters():
#     # w.data += move
#     # loss2 = regressor.forward(X, Y)
#     # # assert np.abs((loss2[0] - loss1[0] ).data.cpu().numpy() - loss_diff) < 1e-9
#     # # print("loss finite diff ", loss2[0] - loss1[0], " projected loss change ", loss_diff)
#     # print("logistic regression gradient test done!")
#     return X, Y, regressor


# def RunToyBitCenterLogistic():
#     X, Y, regressor = GetToyBitCenterLogistic()
#     X = X.half()
#     Y = Y.half()
#     regressor.cuda() 
#     loss1 = regressor.forward(X, Y)
#     loss_diff = 0.0
#     move = 1e-9
#     loss1.backward()
#     for w in regressor.parameters():
#         loss_diff += torch.sum(w.grad.data * move)
#     for w in regressor.parameters():
#         w.data += move
#     loss2 = regressor.forward(X, Y)
#     # assert np.abs((loss2[0] - loss1[0] ).data.cpu().numpy() - loss_diff) < 1e-9
#     # print("loss finite diff ", loss2[0] - loss1[0], " projected loss change ", loss_diff)
#     print("logistic regression gradient test done!")

def TestBitCenterLinearFuncGradientCheck():
    # test if backward is synced with forward in double mode
    input_lp = torch.randn(35, 24, dtype=torch.double, requires_grad=False).cuda()
    input_delta = torch.randn(35, 24, dtype=torch.double, requires_grad=True).cuda()
    weight_lp = torch.randn(17, 24, dtype=torch.double, requires_grad=False).cuda()
    weight_delta = torch.randn(17, 24, dtype=torch.double, requires_grad=True).cuda()
    bias_lp = torch.randn(17, dtype=torch.float32, requires_grad=False).cuda()
    bias_delta = torch.randn(17, dtype=torch.float32, requires_grad=True).cuda()
    bias_lp = None
    bias_delta = None

    inputs = (input_lp, input_delta, weight_lp, weight_delta, bias_lp, bias_delta)
    test = gradcheck(bit_center_linear, inputs, eps=1e-6, atol=1e-4)
    assert test
    print("Bit centering linear function test passed!")


def CheckLayerTensorProperty(t_list):
    # each element of t_list is a tuple containing (t, dtype, is_cuda)
    def CheckSingleTensor(t, dtype, is_cuda, requires_grad):
        assert t.dtype == dtype
        assert t.is_cuda == is_cuda
        assert t.requires_grad == requires_grad
    for i, (t, dtype, is_cuda, requires_grad) in enumerate(t_list):
        if t is None:
            continue
        CheckSingleTensor(t, dtype, is_cuda, requires_grad)


def CheckBitCenterLinearBaseTensorProperty(layer):
    t_list = [(layer.weight, torch.float32, True, True), 
              (layer.bias, torch.float32, True, True),
              (layer.weight_delta, torch.half, True, True),
              (layer.bias_delta, torch.half, True, True),
              (layer.weight_lp, torch.half, True, False),
              (layer.bias_lp, torch.half, True, False),
              (layer.input_cache, torch.half, False, False)]
    CheckLayerTensorProperty(t_list)


def TestBitCenterLinearLayerFwCheck(cast_func=void_cast_func):
    # check if the behavior of BitCentering linear layer is going as expected for forward
    # the backward behavior is guaranteed by 
    # bit center linear function test TestBitCenterLinearFuncGradientCheck
    n_sample = 77
    n_dim = 13
    n_out_dim = 31
    n_minibatch = 3
    minibatch_size = int(n_sample / float(n_minibatch) + 1.0)
    use_bias = True
    layer = BitCenterLinear(in_features=n_dim, out_features=n_out_dim, 
        bias=use_bias, cast_func=cast_func, n_train_sample=n_sample)
    layer.set_mode(do_offset=True)
    layer.cuda()
    input_tensor_list = []
    if (cast_func == single_to_half_det) or (cast_func == single_to_half_stoc): 
        CheckBitCenterLinearBaseTensorProperty(layer)
    for i in range(n_minibatch):
        start_idx = i * minibatch_size
        end_idx = min((i + 1) * minibatch_size, n_sample)
        input_tensor = torch.randn(end_idx - start_idx, n_dim, dtype=torch.float32, requires_grad=True).cuda()        
        output = layer.forward(input_tensor)
        input_tensor_list.append(input_tensor)
        torch.sum(output).backward()
    layer.set_mode(do_offset=False)
    if (cast_func == single_to_half_det) or (cast_func == single_to_half_stoc): 
        CheckBitCenterLinearBaseTensorProperty(layer)
    for i in range(n_minibatch):
        input_lp = layer.input_cache[layer.cache_iter:(layer.cache_iter + input_tensor_list[i].size(0))].cuda()
        output = layer.forward(cast_func(input_tensor_list[i]))
        torch.sum(output).backward()
        output_ref = torch.mm(cast_func(input_tensor_list[i]), layer.weight_lp.t())\
            + torch.mm((input_lp + cast_func(input_tensor_list[i])), layer.weight_delta.t())    
        # print("inter type ", layer.weight_lp.t().data.cpu().numpy().dtype)    
        # output_ref = np.dot(input_tensor_list[i].data.cpu().numpy().astype(np.float16), layer.weight_lp.t().data.cpu().numpy().astype(np.float16))\
        #     + np.dot((input_lp.data.cpu().numpy().astype(np.float16) + input_tensor_list[i].data.cpu().numpy().astype(np.float16)), layer.weight_delta.t().data.cpu().numpy().astype(np.float16))
        if use_bias:
            output_ref += (layer.bias_delta + layer.bias_lp).unsqueeze(0).expand_as(output_ref)
            # output_ref += (layer.bias_delta.data.cpu().numpy().astype(np.float16) + layer.bias_lp.data.cpu().numpy().astype(np.float16))
        output_np = output.cpu().data.numpy()
        # output_ref_np = output_ref
        output_ref_np = output_ref.cpu().data.numpy()
        np.testing.assert_array_almost_equal(output_np, output_ref_np)
    if (cast_func == single_to_half_det) or (cast_func == single_to_half_stoc): 
        CheckBitCenterLinearBaseTensorProperty(layer)
    print("Bit centering linear layer test passed!")


# test if the newly programmed layer is working properly under a sum of output loss
def GetLossFromBitCenterLinear(use_bias=False):
    pass


def TestForwardAndBackwardLinear():
    pass


if __name__ == "__main__":
    print(torch.__version__)
    TestBitCenterLinearLayerFwCheck(cast_func=void_cast_func)
    TestBitCenterLinearLayerFwCheck(cast_func=single_to_half_det)
    TestBitCenterLinearFuncGradientCheck()