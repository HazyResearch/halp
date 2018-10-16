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


def TestBitCenterLinearLayerFloatTensorCheck():
    # check if the behavior of BitCentering linear layer is going as expected
    n_sample = 77
    n_dim = 13
    n_out_dim = 31
    n_minibatch = 3
    minibatch_size = int(n_sample / float(n_minibatch) + 1.0)
    use_bias = True
    layer = BitCenterLinear(in_features=n_dim, out_features=n_out_dim, 
        bias=use_bias, cast_func=void_cast_func, n_train_sample=n_sample)
    layer.set_mode(do_offset=True)
    layer.cuda()
    input_tensor_list = []
    for i in range(n_minibatch):
        start_idx = i * minibatch_size
        end_idx = min((i + 1) * minibatch_size, n_sample)
        input_tensor = torch.randn(end_idx - start_idx, n_dim, dtype=torch.float32, requires_grad=True).cuda()
        _ = layer.forward(input_tensor)
        input_tensor_list.append(input_tensor)
    layer.set_mode(do_offset=False)
    for i in range(n_minibatch):
        input_lp = layer.input_cache[layer.cache_iter:(layer.cache_iter + input_tensor_list[i].size(0))].cuda()
        output = layer.forward(input_tensor_list[i])
        output_ref = torch.mm(input_tensor_list[i], layer.weight_lp.t())\
            + torch.mm((input_lp + input_tensor_list[i]), layer.weight_delta.t())
        if use_bias:
            output_ref += (layer.bias_delta + layer.bias_lp).unsqueeze(0).expand_as(output_ref)
        output_np = output.cpu().data.numpy()
        output_ref_np = output_ref.cpu().data.numpy()
        np.testing.assert_array_almost_equal(output_np, output_ref_np)
    print("Bit centering linear layer test passed!")


# test if the newly programmed layer is working properly under a sum of output loss
def GetLossFromBitCenterLinear(use_bias=False):
    pass


def TestForwardAndBackwardLinear():
    pass


if __name__ == "__main__":
    print(torch.__version__)
    TestBitCenterLinearLayerFloatTensorCheck()
    # TestBitCenterLinearFuncGradientCheck()