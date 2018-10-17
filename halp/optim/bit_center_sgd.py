import torch
import numpy as np
import copy, logging
from torch.autograd import Variable
from torch.optim.optimizer import required
from halp.utils.utils import void_cast_func, single_to_half_det, single_to_half_stoc


class BitCenterOptim(torch.optim.SGD):
	"""
	The base class for bit center optimizer: bit center SGD, bit center SVRG
	"""

	def __init__(self, params, lr=required, weight_decayn_train_sample=1, cast_func=void_cast_func):
		# TODO setup gradient cache using a member function
		# Question: how to deal with the zero gradient thing?
		# Solution 1: remember to clear gradient everytime so that we can add the zero gradient for those not involved parameters
		# Solution 2: distinguish with variable name?
		# TODO Define a fp step function to compute and catch gradient
		# TODO Define a lp step function to compute and update delta
		pass


	def step_lp(self):
		pass


	def step_fp(self):
		pass


class BitCenterSGD(BitCenterOptim):
	pass



class BitCenterSVRG(BitCenterOptim):
	pass