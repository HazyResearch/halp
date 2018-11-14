import os, sys
import math
from copy import deepcopy

# example
#python launch_jobs_nystrom_vs_rff.py census nystrom_vs_rff dawn with_metric cuda -1 run &
#python launch_jobs_nystrom_vs_rff.py covtype nystrom_vs_rff_covtype_no_metric dawn without_metric cuda -1 run &
#for search lamda star for covtype in closeness experiments: python launch_jobs_nystrom_vs_rff.py covtype closeness/classification_real_setting starcluster without_metric cuda 20000 dryrun no_early_stop &
# for sweeping n feat with lambda star for covtype in closeness experiments: python launch_jobs_nystrom_vs_rff_delta.py covtype closeness/classification_real_setting dawn with_metric cuda 20000 dryrun no_early_stop &


def run_experiment(exp_name,
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
                   cluster="starcluster",
                   run_option="dryrun"):
    template = "python /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script/run_models.py " \
               + "--n-epochs=unk " \
               + "--batch-size=unk " \
               + "--reg=unk " \
               + "--alpha=unk " \
               + "--momentum=unk " \
               + "--seed=unk " \
               + " --n-classes=unk " \
               + " --solver=unk " \
               + " --rounding=unk " \
               + " -T=unk " \
               + " --dataset=unk " \
               + " --model=unk " \
               + " --cuda "
    # data_path = "/dfs/scratch0/zjian/float_halp/data/" + experiment_name
    opt = "sgd"
    epoch = 300
    save_path = "/dfs/scratch0/zjian/floating_halp/exp_res/" + exp_name + "/"
    os.system("mkdir -p " + save_path)
    
    template = template.replace("--n-epochs=unk",
                                "--n-epochs=" + str(n_epochs))
    template = template.replace("--batch-size=unk",
                                "--batch-size=" + str(batch_size))
    template = template.replace("--n-classes=unk", "--n-classes=" + str(n_classes))
    template = template.replace("--dataset=unk", "--dataset=" + str(dataset))
    template = template.replace("--model=unk", "--model=" + str(model))

    cnt = 0
    for seed in seed_list:
        for l2_reg in l2_reg_list:
            for lr in lr_list:
                for momentum in momentum_list:
                    for rounding in rounding_list:
                        for T in T_list:
                            for opt in opt_algo_list:
                                save_suffix = "opt_" + opt + "_momentum_" + str(momentum) + "_lr_" + str(
                                    lr) + "_l2_reg_" + str(l2_reg) + "_seed_" + str(seed)
                                command = deepcopy(template)
                                command = command.replace(
                                    "--seed=unk", "--seed=" + str(seed))
                                command = command.replace(
                                    "--reg=unk", "--reg=" + str(l2_reg))
                                command = command.replace(
                                    "--alpha=unk", "--alpha=" + str(lr))
                                command = command.replace(
                                    "--momentum=unk", "--momentum=" + str(momentum))
                                command = command.replace(
                                    "--rounding=unk", "--rounding=" + rounding)
                                command = command.replace(
                                    "-T=unk", "-T=" + str(T))
                                command = command.replace(
                                    "--solver=unk", "--solver=" + opt)
                                os.system("mkdir -p " + save_path + save_suffix)
                                command = "cd /dfs/scratch0/zjian/floating_halp/halp/halp/exp_script && " + command
                                f = open(save_path + save_suffix + "/job.sh", "w")
                                f.write(command)
                                f.close()
                                if cluster == "starcluster":
                                    launch_command = "qsub -V " \
                                    + " -o " + save_path + save_suffix + "/run.log " \
                                    + " -e " + save_path + save_suffix + "/run.err " + save_path + save_suffix + "/job.sh"
                                else:
                                    launch_command = "bash " + save_path + save_suffix + "/job.sh 2>&1 | tee " + save_path + save_suffix + "/run.log"
                                if run_option == "dryrun":
                                    print(launch_command)
                                else:
                                    print(launch_command)
                                    os.system(launch_command)

                                cnt += 1
                    #exit(0)
    print(cnt, "jobs submitted!")


if __name__ == "__main__":
    exp_name = "lenet_hyper_sweep_2018_nov_13"
    n_epochs = 100
    # n_epochs = 2
    batch_size = 128
    n_classes = 10
    l2_reg_list = [5e-4]
    lr_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    # momentum_list = [0.9]
    # seed_list = [1,]
    seed_list = [1, 2, 3]
    # lr_list = [0.01, ]
    momentum_list = [0.9, 0.0]
    # seed_list = [1,]
    opt_algo_list = ["bc-svrg", "bc-sgd", "sgd", "svrg", "lp-sgd", "lp-svrg"]
    # opt_algo_list = ["bc-svrg", "svrg"]
    # opt_algo_list = ["bc-sgd", "sgd"]
    # opt_algo_list = ["lp-sgd", "lp-svrg"]

    rounding_list = ["near"]
    T_list = [391]
    dataset = "cifar10"
    model = "lenet"
    run_option = "dryrun"
    # run_option = "run"
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
        cluster="dawn",
        # cluster="starcluster",
        run_option=run_option)
