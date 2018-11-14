import numpy as np 
import os


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def filter_directory_names(dir_list, pattern_list):
	filtered_list = []
	for dir in dir_list:
		selected=True
		for pattern in pattern_list:
			if pattern not in dir:
				selected = False
				break
		if selected:
			filtered_list.append(dir)
	return filtered_list


def get_results(file_name):
	test_acc = []
	with open(file_name, "r") as f:
		for line in f.readlines():
			if "Test metric" in line:
				val = line.split("acc: ")[1].split(" ")[0]
				test_acc.append(float(val))
	return test_acc


def get_config_with_best_test_acc(top_directory, dir_list):
	best_test_acc = 0.0
	best_config = ""
	best_acc_epoch_id = 0
	for dir in dir_list:
		if not os.path.exists(top_directory + "/" + dir + "/run.log"):
			# print(top_directory + "/" + dir + "/run.log missing!" )
			continue
		acc = get_results(top_directory + "/" + dir + "/run.log")
		if len(acc) == 0:
			continue
		if np.max(acc) > best_test_acc:
			best_test_acc = np.max(acc)
			best_config = dir
			best_acc_epoch_id = np.argmax(acc)
	print("best test acc and config ", best_test_acc, best_acc_epoch_id, best_config)


if __name__ == "__main__":
	top_directory = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_12/"
	all_directories = get_immediate_subdirectories(top_directory)


	pattern_list = ["momentum_0.9", "_bc-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_bc-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_lp-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_lp-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

    #########################################################

	pattern_list = ["momentum_0.9", "_bc-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_bc-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_lp-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_lp-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)


	# file_name = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_12/opt_bc-svrg_momentum_0.0_lr_0.01_l2_reg_0.0005_seed_1/run.log"
	# print(file_name)
	# res = get_results(file_name)
	# print(res)
	# print(np.max(res))