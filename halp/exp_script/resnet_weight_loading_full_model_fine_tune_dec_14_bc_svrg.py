import os, sys
import math
from copy import deepcopy
from halp.utils.launching_utils import run_experiment

if __name__ == "__main__":
    exp_name = "resnet_weight_loading_full_model_fine_tune_dec_14_bc_svrg.py"
    ckpt_path = "resnet_weight_saving_nov_30/opt_lp-sgd_momentum_0.9_lr_0.1_l2_reg_0.0005_seed_unk"
    n_epochs = 100 # to resume from 300
    # n_epochs = 25 # to resume from 325
    batch_size = 128
    n_classes = 10
    l2_reg_list = [5e-4]
    lr_list = [0.01, 0.1, 0.001, 1.0]
    seed_list = [1, 2, 3,]
    momentum_list = [0.0, 0.9]
    T_list = [391]
    dataset = "cifar10"
    model = "resnet"
    run_option = "run"

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
        #cluster="dawn",
        cluster="starcluster",
        run_option=run_option,
        resnet_load_ckpt=True,
        # resnet_load_ckpt_epoch_id=325,
        resnet_load_ckpt_epoch_id=300,
        ckpt_path=ckpt_path,
        resnet_fine_tune=False,
        on_site_compute=True)
