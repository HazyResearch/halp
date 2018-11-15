import numpy as np 
import os

iter_thresh = 391 * 100
epoch_thresh = 100

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_subdirectories_patterns_without_seed(dir_list):
	pattern_list = [x.split("seed_")[0] for x in dir_list]
	pattern_set = list(set(pattern_list))
	return pattern_set

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

def get_test_acc(file_name):
	test_acc = []
	with open(file_name, "r") as f:
		for line in f.readlines():
			if "Test metric" in line:
				val = line.split("acc: ")[1].split(" ")[0]
				try:
					test_acc.append(float(val))
				except:
					print("test acc problematic value ", val)
	return test_acc

def get_train_loss(file_name):
	train_loss = []
	with open(file_name, "r") as f:
		for line in f.readlines():
			if "train loss epoch" in line:
				#val = line.split("loss:")[1].split(" ")[0]
				#print("test ", val)
				try:
					val = line.split("loss:")[1].split(" ")[0]
					train_loss.append(float(val))
				except:
                                        val = line.split("loss: ")[1].split(" ")[0]
                                        train_loss.append(float(val))
	return train_loss

def get_grad_norm(file_name):
	grad_norm = []
	with open(file_name, "r") as f:
		for line in f.readlines():
			if "train loss epoch" in line:
				val = line.split("grad_norm: ")[1].split(" ")[0]
				grad_norm.append(float(val))
	return grad_norm

def get_test_loss(file_name):
	test_loss = []
	with open(file_name, "r") as f:
		for line in f.readlines():
			if "Test metric" in line:
				val = line.split("loss: ")[1].split(" ")[0]
				test_loss.append(float(val))
	return test_loss

def get_ave_metric(pattern, top_directory, seed_list=[1,2,3], metric="test_acc"):
	curve = None
	for seed in seed_list:
		dir = pattern + "seed_" + str(seed)
		if not os.path.exists(top_directory + "/" + dir + "/run.log"):
			print(top_directory + "/" + dir + "/run.log missing!" )
			continue
		if metric == "test_acc":
			values = get_test_acc(top_directory + "/" + dir + "/run.log")
		elif metric == "train_loss":
			values = get_train_loss(top_directory + "/" + dir + "/run.log")
		elif metric == "test_loss":
			values = get_test_loss(top_directory + "/" + dir + "/run.log")
		elif metric == "grad_norm":
			values = get_grad_norm(top_directory + "/" + dir + "/run.log")
		else:
			raise Exception(metric + "is not supported!")
		if curve is None:
			curve = np.array(values)
		else:
			curve += np.array(values)
	return curve / float(len(seed_list))


# def get_config_with_best_test_acc(top_directory, dir_list):
# 	best_test_acc = 0.0
# 	best_config = ""
# 	best_acc_epoch_id = 0
# 	for dir in dir_list:
# 		if not os.path.exists(top_directory + "/" + dir + "/run.log"):
# 			# print(top_directory + "/" + dir + "/run.log missing!" )
# 			continue
# 		acc = get_results(top_directory + "/" + dir + "/run.log")
# 		if len(acc) == 0:
# 			continue
# 		if np.max(acc) > best_test_acc:
# 			best_test_acc = np.max(acc)
# 			best_config = dir
# 			best_acc_epoch_id = np.argmax(acc)
# 	print("best test acc and config ", best_test_acc, best_acc_epoch_id, best_config)

def get_config_with_best_test_acc(top_directory, pattern_list, seed_list=[1, 2, 3]):
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
		ave_acc = ave_acc[:epoch_thresh]
		if np.max(ave_acc) > best_test_acc:
			best_test_acc = np.max(ave_acc)
			best_config = pattern
			best_acc_epoch_id = np.argmax(ave_acc)
	print("best test acc and config ", best_test_acc, best_acc_epoch_id, best_config)

def get_config_with_best_train_loss(top_directory, pattern_list, seed_list=[1, 2, 3]):
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
                    ave_loss = ave_loss[:iter_thresh]
                    ave_loss = running_mean(ave_loss, N=100)		
                    #print(pattern, np.min(ave_loss))
                    if np.min(ave_loss) <  best_train_loss:
                            best_train_loss = np.min(ave_loss)
                            best_config = pattern
                            best_loss_epoch_id = np.argmin(ave_loss)
                except:
                        continue
        print("best train loss and config ", best_train_loss, best_loss_epoch_id, best_config)


if __name__ == "__main__":
	top_directory = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_13/"
	all_directories = get_immediate_subdirectories(top_directory)
	all_directories = get_subdirectories_patterns_without_seed(all_directories)

	pattern_list = ["momentum_0.9", "_bc-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_bc-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_lp-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_lp-svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_svrg"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)
    #########################################################

	pattern_list = ["momentum_0.9", "_bc-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_bc-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_lp-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_lp-sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.9", "_sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	pattern_list = ["momentum_0.0", "_sgd"]
	print("\n")
	print(pattern_list)
	dir_list = filter_directory_names(all_directories, pattern_list)
	get_config_with_best_test_acc(top_directory, dir_list)
	get_config_with_best_train_loss(top_directory, dir_list)

	# file_name = "/dfs/scratch0/zjian/floating_halp/exp_res/lenet_hyper_sweep_2018_nov_12/opt_bc-svrg_momentum_0.0_lr_0.01_l2_reg_0.0005_seed_1/run.log"
	# print(file_name)
	# res = get_results(file_name)
	# print(res)
	# print(np.max(res))
