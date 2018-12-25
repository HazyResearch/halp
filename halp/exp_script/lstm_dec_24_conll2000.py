import os, sys
import math
from copy import deepcopy
from halp.utils.launching_utils import run_experiment

if __name__ == "__main__":
    exp_name = "lstm_dec_24_conll2000"
    n_epochs = 100 # to resume from 300
    # n_epochs = 25 # to resume from 325
    batch_size = 16
    n_classes = 12
    l2_reg_list = [0.0, 1e-5, 1e-4, 1e-3]
    lr_list = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    # seed_list = [1, 2, 3,]
    momentum_list = [0.0, 0.9]
    T_list = [279]
    dataset = "treebank"
    model = "lstm"
    run_option = "run"

    for seed_list in [[1], [2], [3]]:
        opt_algo_list = ["lp-svrg", "lp-sgd"]
        rounding_list = ["near"]    
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
            on_site_compute=True)

        opt_algo_list = ["svrg", "sgd"]
        rounding_list = ["void"]    
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
            # cluster="dawn",
            cluster="starcluster",
            run_option=run_option,
            on_site_compute=True)

        opt_algo_list = ["bc-svrg"]
        rounding_list = ["near"]    
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
            # cluster="dawn",
            cluster="starcluster",
            run_option=run_option,
            on_site_compute=True)
