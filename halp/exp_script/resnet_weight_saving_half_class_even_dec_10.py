import os, sys
import math
from copy import deepcopy
from halp.utils.launching_utils import run_experiment

if __name__ == "__main__":
    exp_name = "resnet_weight_saving_half_class_even_dec_10"
    n_epochs = 350
    # n_epochs = 2
    batch_size = 128
    n_classes = 10
    l2_reg_list = [5e-4]
    lr_list = [0.1]
    # momentum_list = [0.9]
    # seed_list = [1,]
    seed_list = [1, 2, 3]
    # lr_list = [0.01, ]
    momentum_list = [0.9,]
    # seed_list = [1,]
    opt_algo_list = ["lp-sgd"]
    # opt_algo_list = ["sgd"]

    rounding_list = ["near"]
    T_list = [391]
    dataset = "cifar10"
    model = "resnet"
    #run_option = "dryrun"
    run_option = "run"
    run_experiment(
        exp_name,
        n_epochs,
        batch_size,
        n_classes,
        l2_reg_list,
        lr_list,
        momentum_list,
        seed_list,
        opt_algo_list,
        rounding_list,
        T_list,
        dataset,
        model,
        #cluster="dawn",
        cluster="starcluster",
        run_option=run_option,
        resnet_save_ckpt=True,
        only_even_class=True)
