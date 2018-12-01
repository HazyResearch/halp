import numpy as np
import os, sys
import _pickle as cp
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append("../utils/")
# from res_processing_utils import get_ave_metric, running_mean
from res_processing_utils import *
import pandas as pd

def save_csv_only_y(data_list, file_name="./test/test.xlsx", ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    df_list = []
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
#         df_list.append(pd.DataFrame(np.reshape(x, [x.size, 1] ), columns = [label] ) )
        df_list.append(pd.DataFrame(np.reshape(y, [y.size, 1] ), columns = [label] ) )
    pd.concat(df_list, axis=1).to_csv(file_name)
    

def plot_best_config_multiple_epochs(ckpt_epochs, total_epoch=100, win_width=1000, top_directory = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_17/", epoch_len=391):
    pattern_list_array = [ ["_bc-svrg"], ["_lp-svrg"], ["_svrg"], ["_bc-sgd"], ["_lp-sgd"], ["_sgd"]]
    plot_label_list = ["halp", "fp16 lp-svrg", "fp32 svrg", "fp16 bc-sgd", "fp16 lp-sgd", "fp32 sgd"]
#     pattern_list_array = [ ["_bc-svrg"], ["_lp-svrg"]]
#     plot_label_list = ["halp", "fp16 lp-svrg"]
#     top_directory = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_17/"
    all_directories = get_immediate_subdirectories(top_directory)
    all_directories = get_subdirectories_patterns_without_seed(all_directories)

    best_train_loss_list = []
    best_test_acc_list = []
    best_train_loss_config_list = []
    best_test_acc_config_list = []
    for pattern_list in pattern_list_array:
        best_train_loss_list_single_type = []
        best_test_acc_list_single_type = []
        best_train_loss_config_list_single_type = []
        best_test_acc_config_list_single_type = []
        for ckpt_epoch in ckpt_epochs:
            print("\n")
            print(pattern_list, ckpt_epoch)
            dir_list = filter_directory_names(all_directories, pattern_list)
            res = get_config_with_best_test_acc(top_directory, dir_list, cut_off_epoch=ckpt_epoch, total_epoch=total_epoch)
            best_test_acc_list_single_type.append(res[2])
            best_test_acc_config_list_single_type.append(res[1])
            res = get_config_with_best_train_loss(top_directory, dir_list, win_width=win_width, cut_off_epoch=ckpt_epoch, total_epoch=total_epoch, epoch_len=epoch_len)
            best_train_loss_list_single_type.append(res[2])
            best_train_loss_config_list_single_type.append(res[1])
        best_test_acc_list.append(np.array(best_test_acc_list_single_type))
        best_train_loss_list.append(np.array(best_train_loss_list_single_type))

    plot_test_acc(pattern_list_array, best_test_acc_list, x=ckpt_epochs)
    plot_train_loss(pattern_list_array, best_train_loss_list)
    return ckpt_epochs, best_train_loss_list, best_test_acc_list 


def plot_best_config_fixed_epochs(cut_off_epoch=100, total_epoch=100, win_width=1000, top_directory = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_17/", epoch_len=391):
    pattern_list_array = [ ["_bc-svrg"], ["_lp-svrg"], ["_svrg"], ["_bc-sgd"], ["_lp-sgd"], ["_sgd"]]
    plot_label_list = ["halp", "fp16 lp-svrg", "fp32 svrg", "fp16 bc-sgd", "fp16 lp-sgd", "fp32 sgd"]
#     top_directory = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_17/"
    all_directories = get_immediate_subdirectories(top_directory)
    all_directories = get_subdirectories_patterns_without_seed(all_directories)

    best_train_loss_list = []
    best_test_acc_list = []
    best_train_loss_config_list = []
    best_test_acc_config_list = []
    for pattern_list in pattern_list_array:
        print("\n")
        print(pattern_list)
        dir_list = filter_directory_names(all_directories, pattern_list)
        res = get_config_with_best_test_acc(top_directory, dir_list, cut_off_epoch=cut_off_epoch, total_epoch=total_epoch)
        best_test_acc_list.append(res[0])
        best_test_acc_config_list.append(res[1])
        res = get_config_with_best_train_loss(top_directory, dir_list, win_width=win_width, cut_off_epoch=cut_off_epoch, total_epoch=total_epoch, epoch_len=epoch_len)
        best_train_loss_list.append(res[0])
        best_train_loss_config_list.append(res[1])

    plot_test_acc(pattern_list_array, best_test_acc_list)
    plot_train_loss(pattern_list_array, best_train_loss_list)
    return best_test_acc_list, best_train_loss_list, plot_label_list


def plot_test_acc(label_list, test_acc_list, x=None):
    plt.figure()
    data_list = []
    for test_acc, label in zip(test_acc_list, label_list):
        data_list.append((label, np.arange(test_acc.size) + 1, test_acc))
        if x is not None:
            plt.plot(x, test_acc, label=label)
        else:
            plt.plot(test_acc, label=label)
    plt.legend(loc="lower right")
    plt.ylim([0.6, 0.65])
    # plt.xlim([None, 60])
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR10 (LeNet)")
    plt.grid()
    plt.show()
#     save_csv(data_list, file_name="./workspace/lenet_test_acc_all.csv")

    plt.figure()
    data_list = []
    for test_acc, label in zip(test_acc_list, label_list):
        data_list.append((label, np.arange(test_acc.size) + 1, np.maximum.accumulate(test_acc)))
        if x is not None:
            plt.plot(x, np.maximum.accumulate(test_acc), label=label)
        else:
            plt.plot(np.maximum.accumulate(test_acc), label=label)
    plt.legend(loc="lower right")
    plt.ylim([0.6, 0.65])
    # plt.xlim([None, 60])
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR10 (LeNet)")
    plt.grid()
    plt.show()
#     save_csv(data_list, file_name="./workspace/lenet_test_acc_all.csv")


def plot_train_loss(label_list, train_loss_list):
    plt.figure()
    data_list = []
    for train_loss, label in zip(train_loss_list, label_list):
        data_list.append((label, np.arange(train_loss.size) + 1, train_loss))
        plt.plot(train_loss, label=label)
    plt.legend()
    plt.ylim([0.4, 0.8])
    plt.xlabel("Iterations")
    plt.ylabel("Training loss")
    plt.title("CIFAR10 (LeNet)")
    plt.grid()
    plt.show()
#     save_csv(data_list, file_name="./workspace/lenet_train_loss_all.csv")


def get_config_with_best_train_loss(top_directory, pattern_list, seed_list=[1, 2, 3], win_width=1000, cut_off_epoch=100, total_epoch=100, epoch_len=391):
        best_train_loss = np.finfo(dtype=np.float64).max 
        best_config = ""
        best_loss_epoch_id = 0
        for pattern in pattern_list:
                ave_loss = None
                try:
                    for seed in seed_list:
                            #print(pattern, seed)
                            dir = pattern + "seed_" + str(seed)
                            if not os.path.exists(top_directory + "/" + dir + "/run.log"):
                                    print(top_directory + "/" + dir + "/run.log missing!" )
                                    continue
                            loss = get_train_loss(top_directory + "/" + dir + "/run.log")
                            if len(loss) == 0:
                                    print(top_directory + "/" + dir + "/run.log has 0 train loss record" )
                                    continue
                            if ave_loss is None:
                                     ave_loss = np.array(loss)
                            else:
                                     ave_loss += np.array(loss)
                    ave_loss /= len(seed_list)
                    assert cut_off_epoch <= total_epoch
                    iter_thresh = ave_loss.size * cut_off_epoch // total_epoch
                    ave_loss = ave_loss[:iter_thresh]
                    ave_loss_test = running_mean(ave_loss, N=win_width)
                    ave_loss = np.reshape(ave_loss, (-1, epoch_len))
#                     print(ave_loss, ave_loss_test)
                    ave_loss = np.mean(ave_loss, axis=1).reshape((ave_loss.shape[0], ))
#                     print(ave_loss)
                    #print(pattern, np.min(ave_loss))
                    if np.min(ave_loss) <  best_train_loss:
                            best_train_loss = np.min(ave_loss)
                            best_train_loss_full = ave_loss
                            best_config = pattern
                            best_loss_epoch_id = np.argmin(ave_loss)
                except:
                        continue
        print("best train loss and config ", best_train_loss, best_loss_epoch_id, best_config)
        return (best_train_loss_full, best_loss_epoch_id, best_train_loss, best_config)
    

def get_config_with_best_test_acc(top_directory, pattern_list, seed_list=[1, 2, 3], cut_off_epoch=100, total_epoch=100):
    best_test_acc = 0.0
    best_config = ""
    best_acc_epoch_id = 0
    for pattern in pattern_list:
        ave_acc = None
        for seed in seed_list:
            dir = pattern + "seed_" + str(seed)
            if not os.path.exists(top_directory + "/" + dir + "/run.log"):
                print(top_directory + "/" + dir + "/run.log missing!" )
                continue
            acc = get_test_acc(top_directory + "/" + dir + "/run.log")
            if len(acc) == 0:
                print(top_directory + "/" + dir + "/run.log has 0 test accuracy record" )
                continue
            if ave_acc is None:
                ave_acc = np.array(acc)
            else:
                ave_acc += np.array(acc)
        ave_acc /= len(seed_list)
        assert cut_off_epoch <= total_epoch
        ave_acc = ave_acc[:cut_off_epoch]
        if np.max(ave_acc) > best_test_acc:
            best_test_acc = np.max(ave_acc)
            best_test_acc_full = ave_acc
            best_config = pattern
            best_acc_epoch_id = np.argmax(ave_acc)
    print("best test acc and config ", best_test_acc, best_acc_epoch_id, best_config)
    return (best_test_acc_full, best_acc_epoch_id, best_test_acc, best_config)


def save_csv_with_error_bar(data_list, file_name="./test/test.csv", ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    df_list = []
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
        average_y = average_results_array(y)
        std_y = std_results_array(y)
        if ave_x:
            x = average_results_array(x)
        x = np.array(x)
        if len(x.shape) == 2:
            n_pt = x.shape[1]
            x = np.mean(x, axis=0).reshape((n_pt, ) )
        average_y = np.array(average_y)
        std_y = np.array(std_y)
        assert x.shape == average_y.shape
        assert x.shape == std_y.shape
        df_list.append(pd.DataFrame(np.reshape(x, [x.size, 1] ), columns = [label + "|x" ] ) )
        df_list.append(pd.DataFrame(np.reshape(average_y, [average_y.size, 1] ), columns = [label + "|y" ] ) )
        df_list.append(pd.DataFrame(np.reshape(std_y, [std_y.size, 1] ), columns = [label + "|y_std" ] ) )
    pd.concat(df_list, axis=1).to_csv(file_name)
    
    
def save_csv(data_list, file_name="./test/test.csv", ave_x=False):
    '''
    data is a list of tuple (label, x_pt, y_pt), it is plotted using color named as label in the color_dict.
    x_pt is a 1d list, y_pt is list of list, each inner list is from a random seed.
    '''
    df_list = []
    for i in range(len(data_list) ):
        label = data_list[i][0]
        x = data_list[i][1]
        y = data_list[i][2]
        df_list.append(pd.DataFrame(np.reshape(x, [x.size, 1] ), columns = [label + "|x" ] ) )
        df_list.append(pd.DataFrame(np.reshape(y, [y.size, 1] ), columns = [label + "|y" ] ) )
    pd.concat(df_list, axis=1).to_csv(file_name)