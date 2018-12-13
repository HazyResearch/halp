import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from halp.utils.utils import LP_DEBUG_EPOCH_LEN, DOUBLE_PREC_DEBUG_EPOCH_LEN
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('')


def get_partial_classes(dataset, train=True, test_func=lambda x: x % 2 == 0, label_transform=lambda x: x // 2):
	if train:
		labels_all_classes = dataset.train_labels
	else:
		labels_all_classes = dataset.test_labels
	idx_labels = [(i, label) for i, label in enumerate(labels_all_classes) if test_func(label)]
	idx, labels = zip(*idx_labels)
	if train:
		dataset.train_data = dataset.train_data[idx, :, :, :]
		dataset.train_labels = labels
		old_label = list(dataset.train_labels).copy()
		dataset.train_labels = [label_transform(x) for x in dataset.train_labels]
		new_label = dataset.train_labels.copy()
	else:
		dataset.test_data = dataset.test_data[idx, :, :, :]
		dataset.test_labels = labels
		old_label = list(dataset.test_labels).copy()
		dataset.test_labels = [label_transform(x) for x in dataset.test_labels]
		new_label = dataset.test_labels.copy()

	logger.info("n samples after sample even classes: " + str(len(new_label)))
	logger.info("label classes before even classes extraction" + np.array2string(np.unique(old_label)))
	logger.info("label classes after even classes extraction" + np.array2string(np.unique(new_label)))
	return dataset

def get_cifar10_data_loader(batch_size=128, args=None):
	print('==> Preparing data..')
	LP_DEBUG = args.float_debug
	DOUBLE_PREC_DEBUG = args.double_debug
	# transform_train = transforms.Compose([
	#     transforms.RandomCrop(32, padding=4),
	#     transforms.RandomHorizontalFlip(),
	#     transforms.ToTensor(),
	#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# ])

	# transform_test = transforms.Compose([
	#     transforms.ToTensor(),
	#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	# ])
	
	transform_train = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])


	# X_train = torch.randn(128 * 3, 3, 32, 32, dtype=torch.double)
	# Y_train = torch.LongTensor(128 * 3).random_(10)
	# trainset = torch.utils.data.TensorDataset(X_train, Y_train)

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	if DOUBLE_PREC_DEBUG:
		trainset = torch.utils.data.Subset(trainset, np.arange(batch_size * DOUBLE_PREC_DEBUG_EPOCH_LEN))
	elif LP_DEBUG:
		trainset = torch.utils.data.Subset(trainset, np.arange(batch_size * LP_DEBUG_EPOCH_LEN))
	if args.only_even_class:
		get_partial_classes(trainset, 
							train=True, 
							test_func=lambda x: x % 2 == 0, 
							label_transform=lambda x: x // 2)
		args.n_classes = (args.n_classes + 1) // 2
		logger.info("Data stat after extracting even classes: n_class=" + str(args.n_classes))
	elif args.only_odd_class:
		get_partial_classes(trainset, 
							train=True, 
							test_func=lambda x: x % 2 == 1, 
							label_transform=lambda x: (x - 1) // 2)
		args.n_classes = args.n_classes // 2
		logger.info("Data stat after extracting odd classes: n_class=" + str(args.n_classes))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)
	args.T = len(trainloader)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	test_batch_size = 100
	if DOUBLE_PREC_DEBUG:
		testset = torch.utils.data.Subset(testset, np.arange(batch_size * DOUBLE_PREC_DEBUG_EPOCH_LEN))
		test_batch_size = args.batch_size
	elif LP_DEBUG:
		testset = torch.utils.data.Subset(testset, np.arange(batch_size * LP_DEBUG_EPOCH_LEN))
		test_batch_size = args.batch_size	
	if args.only_even_class:
		testset = get_partial_classes(testset, 
							train=False, 
							test_func=lambda x: x % 2 == 0, 
							label_transform=lambda x: x // 2)
	elif args.only_odd_class:
		testset = get_partial_classes(testset, 
							train=False, 
							test_func=lambda x: x % 2 == 1, 
							label_transform=lambda x: (x - 1) // 2)

	testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=1)
	
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	input_shape = (batch_size, 3, 32, 32) 
	return trainloader, testloader, input_shape, len(trainset)