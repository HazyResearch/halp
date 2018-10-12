import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
from halp.utils.utils import single_to_half_det, single_to_half_stoc


class BitCenterLinear(nn.Linear):
    # TODO: consider to sanity check with 32 bit delta terms
    def __init__(self, in_features, out_features, bias=True, cast_func=single_to_half_det):
        super(BitCenterLinear, self).__init__(in_features, out_features, bias)
        self.cast_func = cast_func
        # TODO check if this is a parameter after casting
        # weight_delta is the delta tensor in the algorithm while weight_lp is the cached 
        # lp version of weight
        # TODO make weight_lp bias_lp all in no-gradient mode
        # TODO check if weight delta is with gradient
        self.weight_delta = self.cast_func(self.weight)
        self.weight_lp = Parameter(self.cast_func(self.weight.data), require_grad=False)
        self.do_offset = False
        if bias:
            self.bias_delta = self.cast_func(self.bias)
            self.bias_lp = Parameter(self.cast_func(self.bias.data), require_grad=False)
        else:
            self.register_parameter('bias_delta', None)
            self.register_parameter('bias_lp', None)
        # pre allocate memory for the weight and bias converted to lp
        self.reset_parameters_bit_center()

    def reset_parameters_bit_center(self):
        init.zero_(self.weight_delta)
        if self.bias is not None:
            init.zero_(self.bias_delta)

    def set_mode(do_offset):
        self.do_offset = offset

    # TODO: Test whether forward_fp properly generate forward output and backward output
    # Consider how to adapt to LP SGD and LP SVRG mode
    # def forward_fp(self, input):
    #     return F.linear(input, self.weight, self.bias)

    def forward(self, input):
        if self.do_offset:
            self.input_lp = Parameter(self.cast_func(input.data), require_grad=False)
            return F.linear(input, self.weight, self.bias)
        else:
            # For here input is delta_x
            return self.weight_lp * input + (self.input_lp + input) * self.weight_delta


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


# if __name__ == "__main__":
#     RunToyBitCenterLogistic()
        
