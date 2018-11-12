import torch
import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')

def evaluate_acc(model, val_loader, use_cuda=True, dtype="fp"):
    model.eval()
    correct_cnt = 0
    sample_cnt = 0
    cross_entropy_accum = 0.0
    for i, (X, Y) in enumerate(val_loader):
        if use_cuda:
            X, Y = X.cuda(), Y.cuda()
        if dtype == "lp":
            X = model.cast_func(X)
        # if len(list(X.size())) != 2:
        #     X = X.view(X.size(0), -1)
        pred, output = model.predict(X)
        assert pred.shape == Y.data.cpu().numpy().shape
        correct_cnt += np.sum(pred == Y.data.cpu().numpy())
        cross_entropy_accum += model.criterion(
            output, Y).data.cpu().numpy() * X.shape[0]
        sample_cnt += pred.size
    logger.info("Eval acc " + str(correct_cnt / float(sample_cnt)) +
                " eval cross entropy " +
                str(cross_entropy_accum / float(sample_cnt)))
    return (correct_cnt / float(sample_cnt),
            cross_entropy_accum / float(sample_cnt))


def train_non_bit_center_optimizer(model,
                                   optimizer,
                                   train_loader,
                                   val_loader,
                                   n_epochs,
                                   eval_func=evaluate_acc,
                                   use_cuda=True,
                                   dtype='fp'):
    train_loss_list = []
    eval_metric_list = []

    logging.info("using training function for non bit center optimizers")
    if optimizer.T is not None:
        logging.info("optimizer T=" + str(optimizer.T))
    for epoch_id in range(n_epochs):
        model.train()
        for i, (X, Y) in enumerate(train_loader):
            if use_cuda:
                X, Y = X.cuda(), Y.cuda()
            if dtype == "lp":
                X = optimizer.cast_func(X)
            if dtype == "bc":
                raise Exception("This function can only run non-bc optimizers")
            optimizer.zero_grad()
            train_loss = model(X.double(), Y)
            train_loss.backward()
            if optimizer.__class__.__name__ == "SVRG":

                def svrg_closure(data=X, target=Y):
                    if use_cuda:
                        data = data.cuda()
                        target = target.cuda()
                    if dtype == "lp":
                        data = optimizer.cast_func(data)
                    if dtype == "bc":
                        raise Exception("This function can only run non-bc optimizers")
                    loss = model(data, target)
                    loss.backward()
                    return loss

                optimizer.step(svrg_closure)
            else:
                optimizer.step()
            train_loss_list.append(train_loss.item())
            logger.info("train loss e/i/l " + str(epoch_id) + " " + str(i) + " " + str(train_loss.item()))
        logger.info("Finished train epoch " + str(epoch_id))
        model.eval()
        # eval_metric_list.append(eval_func(model, val_loader, use_cuda, dtype))
    return train_loss_list, eval_metric_list


def train_bit_center_optimizer(model,
                               optimizer,
                               train_loader,
                               val_loader,
                               n_epochs,
                               eval_func=evaluate_acc,
                               use_cuda=True,
                               dtype="bc"):
    train_loss_list = []
    eval_metric_list = []
    T = optimizer.T
    total_iter = 0

    logging.info("using training function for bit center optimizers")
    logging.info("optimizer T=" + str(optimizer.T))
    for epoch_id in range(n_epochs):
        model.train()
        for i, (X, Y) in enumerate(train_loader):
            if total_iter % T == 0:
                optimizer.on_start_fp_steps(model)
                for j, (X_fp, Y_fp) in enumerate(train_loader):
                    optimizer.zero_grad()
                    if use_cuda:
                        X_fp, Y_fp = X_fp.cuda(), Y_fp.cuda()
                    loss_fp = model(X_fp.double(), Y_fp)
                    loss_fp.backward()
                    optimizer.step_fp()
                    if j < 3:
                        print("fp steps ", j, loss_fp.item())
                optimizer.on_end_fp_steps(model)
                optimizer.on_start_lp_steps(model)
            if use_cuda:
                X, Y = X.cuda(), Y.cuda()
            # note here X is the input delta. It is suppose to be zero.
            X = optimizer.cast_func(X).zero_()
            if dtype != "bc":
                raise Exception(
                    "This training function does not support dtype other than bc"
                )
            optimizer.zero_grad()
            train_loss = model(X.double(), Y)
            train_loss.backward()
            optimizer.step_lp()
            train_loss_list.append(train_loss.item())
            if total_iter % T == T - 1:
                optimizer.on_end_lp_steps(model)
            total_iter += 1
            logger.info("train loss e/i/l " + str(epoch_id) + " " + str(i) + " " + str(train_loss.item()))
        logger.info("Finished train epoch " + str(epoch_id))
        model.eval()
        optimizer.on_start_fp_steps(model)
        # eval_metric_list.append(eval_func(model, val_loader, use_cuda, dtype=dtype))
        optimizer.on_end_fp_steps(model)
    return train_loss_list, eval_metric_list
