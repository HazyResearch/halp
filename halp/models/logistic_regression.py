import torch
import numpy as np
from torch.autograd import Variable

import halp.utils.utils

from halp.utils.utils import single_to_half_det, single_to_half_stoc, void_cast_func
from halp.layers.bit_center_layer import BitCenterModule
from halp.layers.linear_layer import BitCenterLinear
from halp.layers.cross_entropy import BitCenterCrossEntropy


class LogisticRegression(BitCenterModule):
  def __init__(self, input_dim, n_class, reg_lambda, dtype="bc", 
               cast_func=void_cast_func, n_train_sample=1):
    super(LogisticRegression, self).__init__()
    self.input_dim = input_dim
    self.n_class = n_class
    self.reg_lambda = reg_lambda
    self.linear = \
        BitCenterLinear(self.input_dim, out_features=self.n_class, 
                        cast_func=cast_func, n_train_sample=n_train_sample)
    self.criterion = \
        BitCenterCrossEntropy(cast_func=cast_func,
                              n_train_sample=n_train_sample)

    if dtype == "bc":
        pass    
    elif dtype == "fp":
        # we use the copied weights to guarantee same initialization
        # across different dtype when using the same random seed
        linear_tmp = self.linear
        self.linear = torch.nn.Linear(self.input_dim, out_features=self.n_class)
        self.criterion = torch.nn.CrossEntropyLoss(size_average=True)
        self.linear.weight.data.copy_(linear_tmp.weight)
        self.linear.bias.data.copy_(linear_tmp.bias)
    elif dtype == "lp":
        pass
    else:
        raise Exception("dtype not supported")
    self.dtype = dtype

  def forward(self, x, y):
    self.output = self.linear(x)
    if len(list(y.size() ) ) == 2:
        y = y.squeeze()
    self.loss = self.criterion(self.output, y)
    return self.loss

  def predict(self, x):
    output = self.linear(x)
    if isinstance(self.linear, BitCenterLinear):
        assert self.linear.do_offset == True
    pred = output.data.cpu().numpy().argmax(axis=1)
    return pred, output






