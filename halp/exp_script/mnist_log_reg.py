import copy
import argparse
import numpy as np
import torch
import torch.utils.data
from torch.optim import SGD
from halp.optim.bit_center_sgd import BitCenterSGD
from halp.optim.bit_center_svrg import BitCenterSVRG
from halp.optim.svrg import SVRG
from halp.models.logistic_regression import LogisticRegression
from halp.utils.mnist_data_utils import load_mnist
from halp.utils import utils
from halp.utils.utils import void_cast_func, single_to_half_det
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')


parser = argparse.ArgumentParser()
parser.add_argument("-T", action="store", default=1, type=int,
                    help="T parameter for SVRG type algorithms.")
parser.add_argument("-e", "--n-epochs", action="store", default=10, type=int,
                    help="Number of epochs to run for")
parser.add_argument("-bs", "--batch-size", action="store", default=100, type=int,
                    help="Batch size.")
parser.add_argument("-a", "--alpha", action="store", default=0.01, type=float,
                    help="Learning Rate")
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
parser.add_argument("--solver", action="store", default="all", type=str,
                    choices=["sgd", "svrg", 
                             "lp-sgd", "lp-svrg", 
                             "bc-sgd", "bc-svrg"],
                    help="Solver/optimization algorithm.")
parser.add_argument("--reg", type=float, default=0.0, 
                    help="L2 regularizer strength")
parser.add_argument("--cuda", action="store_true", 
                    help="currently pytorch only support store true")
args = parser.parse_args()

utils.set_seed(args.seed)
X_train, X_val, Y_train, Y_val = load_mnist(onehot=False)
X_train, X_val = torch.FloatTensor(X_train), torch.FloatTensor(X_val)
Y_train, Y_val = torch.LongTensor(Y_train), torch.LongTensor(Y_val)

train_data = \
        torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=False)
val_data = \
    torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.batch_size, shuffle=False)

# determine the dtype
if args.solver.startswith("bc-"):
    args.dtype = "bc"
elif args.solver.startswith("lp-"):
    args.dtype = "lp"
else:
    args.dtype = "fp"

# note reg_lambda is dummy here, the regularizer is handled by the optimizer
model = LogisticRegression(
    input_dim=X_train.shape[1],
    n_class=args.n_classes,
    reg_lambda=args.reg,
    dtype=args.dtype,
    cast_func=single_to_half_det,
    n_train_sample=X_train.shape[0])
if args.cuda:
    model.cuda()

# setup optimizer
params_name = [x for x, y in model.named_parameters()]
params = [y for x, y in model.named_parameters()]

logger.info("Params list: ")
for x in params_name:
    logger.info(x)

if (args.solver == "sgd") or (args.solver == "lp-sgd"):
    optimizer = SGD(params=params, lr=args.alpha, weight_decay=args.reg)
elif args.solver == "svrg" or "lp-svrg":
    optimizer = SVRG(
        params=params,
        lr=args.alpha,
        weight_decay=args.reg,
        T=int(args.T * X_train.size(0)),
        data_loader=train_loader)
elif args.solver == "bc-sgd":
    optimizer = BitCenterSGD(
        params=params,
        params_name=params_name,
        lr=args.alpha,
        weight_decay=args.reg,
        n_train_sample=x_train.size(0),
        cast_func=single_to_half_det,
        minibatch_size=args.batch_size)
elif args.solver == "bc-svrg":
    optimizer = BitCenterSVRG(
        params=params,
        params_name=params_name,
        lr=args.alpha,
        weight_decay=args.reg,
        n_train_sample=x_train.size(0),
        cast_func=single_to_half_det,
        minibatch_size=args.batch_size)
else:
    raise Exception(args.solver + " is an unsupported optimizer.")


def evaluate_acc(model, val_loader, use_cuda=True):
    model.eval()
    correct_cnt = 0
    sample_cnt = 0
    cross_entropy_accum = 0.0
    for i, (X, Y) in enumerate(val_loader):
        if use_cuda:
            X, Y = X.cuda(), Y.cuda()
        pred, output = model.predict(X)
        assert pred.shape == Y.data.cpu().numpy().shape
        correct_cnt += np.sum(pred == Y.data.cpu().numpy())
        cross_entropy_accum += model.criterion(output, Y).data.cpu().numpy() * X.shape[0]
        sample_cnt += pred.size
    logger.info("Eval acc " + str(correct_cnt / float(sample_cnt)) + " eval cross entropy "
          + str(cross_entropy_accum / float(sample_cnt)))
    return (correct_cnt / float(sample_cnt),
            cross_entropy_accum / float(sample_cnt))


def train_non_bit_center_optimizer(model,
                                   optimizer,
                                   train_loader,
                                   val_loader,
                                   n_epochs,
                                   eval_func=evaluate_acc,
                                   use_cuda=True):
    train_loss_list = []
    eval_metric_list = []
    for epoch_id in range(n_epochs):
        model.train()
        for i, (X, Y) in enumerate(train_loader):
            if use_cuda:
                X, Y = X.cuda(), Y.cuda()    
            optimizer.zero_grad()
            train_loss = model(X, Y)
            train_loss.backward()
            if optimizer.__class__.__name__ == "SVRG":
                def svrg_closure(data=X, target=Y):
                    if use_cuda:
                        data = data.cuda()
                        target = target.cuda()
                    loss = model(data, target)
                    loss.backward()
                    return loss
                optimizer.step(svrg_closure)
            else:
                optimizer.step()
            train_loss_list.append(train_loss.item())
            # print(train_loss)
        logger.info("Finished train epoch " + str(epoch_id))
        model.eval()
        eval_metric_list.append(eval_func(model, val_loader, use_cuda))
    return train_loss_list, eval_metric_list


def train_bit_center_optimizer(model, optimizer, train_loader, val_loader,
                               n_epochs):
    pass


# run training procedure
logger.info("optimizer " + optimizer.__class__.__name__)
logger.info("model " + model.linear.__class__.__name__)
if (args.solver == "bc-sgd") or (args.solver == "bc-svrg"):
    train_loss = train_bit_center_optimizer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        use_cuda=args.cuda)
else:
    train_loss = train_non_bit_center_optimizer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        use_cuda=args.cuda)


