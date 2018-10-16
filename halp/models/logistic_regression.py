import torch
import numpy as np
from torch.autograd import Variable

import halp.utils.utils

from halp.utils.utils import single_to_half_det, single_to_half_stoc, void_cast_func
from halp.layers.linear_layer import BitCenterLinear


class LogisticRegression(torch.nn.Module):
  def __init__(self, input_dim, n_class, reg_lambda, dtype="float"):
    super(LogisticRegression, self).__init__()
    self.input_dim = input_dim
    self.n_class = n_class
    self.reg_lambda = reg_lambda
    if dtype == "half":
        self.linear = BitCenterLinear(self.input_dim, out_features=self.n_class, cast_func=single_to_half_det)
    else:
        self.linear = torch.nn.Linear(self.input_dim, out_features=self.n_class)
    self.criterion = torch.nn.CrossEntropyLoss(size_average=True)
    self.dtype = dtype
    if self.dtype == "double":
      for w in self.parameters():
        w.data = w.data.type(torch.DoubleTensor)

  def forward(self, x, y):
    self.output = self.linear(x)
    if len(list(y.size() ) ) == 2:
        y = y.squeeze()
    self.loss = self.criterion(self.output, y)
    return self.loss

  def predict(self, x):
    output = self.linear(x)
    pred = output.data.cpu().numpy().argmax(axis=1)
    return pred, output






