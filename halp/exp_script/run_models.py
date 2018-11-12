import copy
import argparse
import math
import numpy as np
import torch
torch.backends.cudnn.deterministic=True
import torch.nn as nn
import torch.utils.data
from torch.optim import SGD
from halp.optim.bit_center_sgd import BitCenterSGD
from halp.optim.bit_center_svrg import BitCenterSVRG
from halp.optim.svrg import SVRG
from halp.models.logistic_regression import LogisticRegression
from halp.models.lenet import LeNet
from halp.utils.mnist_data_utils import get_mnist_data_loader
from halp.utils.cifar_data_utils import get_cifar10_data_loader
from halp.utils import utils
from halp.utils.utils import void_cast_func
from halp.utils.utils import single_to_half_det, single_to_half_stoc
from halp.utils.train_utils import evaluate_acc
from halp.utils.train_utils import train_non_bit_center_optimizer
from halp.utils.train_utils import train_bit_center_optimizer
from halp.utils.mnist_data_utils import get_mnist_data_loader
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')
import time


parser = argparse.ArgumentParser()
parser.add_argument("-T", action="store", default=1, type=int,
                    help="T parameter for SVRG type algorithms.")
parser.add_argument("-e", "--n-epochs", action="store", default=10, type=int,
                    help="Number of epochs to run for")
parser.add_argument("-bs", "--batch-size", action="store", default=100, type=int,
                    help="Batch size.")
parser.add_argument("-a", "--alpha", action="store", default=0.01, type=float,
                    help="Learning Rate")
parser.add_argument("-m", "--momentum", default=0.0, type=float,
                    help="momentum value for Polyak's momentum algorithm")
parser.add_argument("-b", "--n-bits", action="store", default=8, type=int,
                    help="Number of bits of precision")
parser.add_argument("--lin-fwd-sf", action="store", default=1, type=float,
                    help="Linear layer forward scale factor.")
parser.add_argument("--lin-bck-sf", action="store", default=1e-2, type=float,
                    help="Linear layer backwards scale factor.")
parser.add_argument("--loss-sf", action="store", default=1e-3, type=float,
                    help="Loss scale factor.")
parser.add_argument("-s", "--seed", action="store", default=42, type=int,
                    help="Random seed.")
parser.add_argument("-c", "--n-classes", action="store", default=10, type=int,
                    help="Number of classes for classification.")
parser.add_argument("--solver", action="store", default="sgd", type=str,
                    choices=["sgd", "svrg", 
                             "lp-sgd", "lp-svrg", 
                             "bc-sgd", "bc-svrg"],
                    help="Solver/optimization algorithm.")
parser.add_argument("--reg", type=float, default=0.0, 
                    help="L2 regularizer strength")
parser.add_argument("--cuda", action="store_true", 
                    help="currently pytorch only support store true.")
parser.add_argument("--debug-test", action="store_true",
                    help="switch to use small toy example for debugging.")
parser.add_argument("--rounding", default="near", type=str,
                    choices=["near", "stoc", "void"],
                    help="Support nearest (near) and stochastic (stoc) rounding.")
parser.add_argument("--dataset", default="mnist", type=str, 
                    choices=["mnist", "cifar10"], 
                    help="The dataset to train on.")
parser.add_argument("--model", default="logreg", type=str,
                    choices=["logreg", "lenet"],
                    help="The model used on the given dataset.")
args = parser.parse_args()
utils.set_seed(args.seed)

if args.dataset == "mnist":
    train_loader, val_loader, input_shape, n_train_sample = get_mnist_data_loader(
        onehot=False, debug_test=args.debug_test, batch_size=args.batch_size)
elif args.dataset == "cifar10":
    train_loader, val_loader, input_shape, n_train_sample = get_cifar10_data_loader(
        batch_size=args.batch_size)

if args.debug_test:
    args.cast_func = void_cast_func
    args.T = len(train_loader)
    args.batch_size = 1    
elif args.rounding == "near":
    args.cast_func = single_to_half_det
elif args.rounding == "stoc":
    args.cast_func = single_to_half_stoc
elif args.rounding == "void":
    args.cast_func = void_cast_func
else:
    raise Exception("The rounding method is not supported!")

# TODO resolve this for trainin procedure and avoid this check
print("dataset stats: n_batch, batch_size, T ", len(train_loader), args.batch_size, args.T)
if len(train_loader) != args.T:
    raise Exception("Currently not supporting settings other than T = 1 epoch, please resolve")

# determine the dtype
if args.solver.startswith("bc-"):
    args.dtype = "bc"
elif args.solver.startswith("lp-"):
    args.dtype = "lp"
else:
    args.dtype = "fp"

# note reg_lambda is dummy here, the regularizer is handled by the optimizer
if args.model == "logreg":
    model = LogisticRegression(
        input_dim=input_shape[1],
        n_class=args.n_classes,
        reg_lambda=args.reg,
        dtype=args.dtype,
        cast_func=args.cast_func,
        n_train_sample=n_train_sample)
elif args.model == "lenet":
    model = LeNet(
        cast_func=args.cast_func,
        n_train_sample=n_train_sample,
        dtype=args.dtype)
else:
    raise Exception(args.model + " is currently not supported!")

if args.cuda:
    # note as the cache are set up in the first foward pass
    # the location of the cache is not controled by the cuda() here
    model.cuda()

# setup optimizer
params_name = [x for x, y in model.named_parameters()]
params = [y for x, y in model.named_parameters()]

logger.info("Params list: ")
for name, p in zip(params_name, params):
    logger.info(name + " " + str(p.dtype))
if (args.solver == "sgd") or (args.solver == "lp-sgd"):
    optimizer = SGD(
        params=params,
        lr=args.alpha,
        momentum=args.momentum,
        weight_decay=args.reg)
    optimizer.cast_func = args.cast_func
    optimizer.T = None
elif (args.solver == "svrg") or (args.solver == "lp-svrg"):
    optimizer = SVRG(
        params=params,
        lr=args.alpha,
        momentum=args.momentum,
        weight_decay=args.reg,
        T=args.T,
        data_loader=train_loader)
    optimizer.cast_func = args.cast_func
elif args.solver == "bc-sgd":
    optimizer = BitCenterSGD(
        params=params,
        params_name=params_name,
        lr=args.alpha,
        momentum=args.momentum,
        weight_decay=args.reg,
        n_train_sample=n_train_sample,
        cast_func=args.cast_func,
        minibatch_size=args.batch_size,
        T=args.T)
elif args.solver == "bc-svrg":
    optimizer = BitCenterSVRG(
        params=params,
        params_name=params_name,
        lr=args.alpha,
        momentum=args.momentum,
        weight_decay=args.reg,
        n_train_sample=n_train_sample,
        cast_func=args.cast_func,
        minibatch_size=args.batch_size,
        T=args.T)
else:
    raise Exception(args.solver + " is an unsupported optimizer.")


start_time = time.time()
# run training procedure
logger.info("optimizer " + optimizer.__class__.__name__)
logger.info("model " + model.__class__.__name__)
logger.info("optimizer rounding func " + optimizer.cast_func.__name__)
logger.info("model rounding func " + model.cast_func.__name__)
model.print_module_types()
if (args.solver == "bc-sgd") or (args.solver == "bc-svrg"):
    train_loss = train_bit_center_optimizer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        use_cuda=args.cuda,
        dtype=args.dtype)
else:
    train_loss = train_non_bit_center_optimizer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        use_cuda=args.cuda,
        dtype=args.dtype)
end_time = time.time()
print("Elapsed training time: ", end_time - start_time)
