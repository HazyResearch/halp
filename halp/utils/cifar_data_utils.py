import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_data_loader(batch_size=128):
	print('==> Preparing data..')
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
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	input_shape = (batch_size, 3, 32, 32) 
	return trainloader, testloader, input_shape, len(trainset)