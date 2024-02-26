import torch

upper_limit, lower_limit = 1, 0

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

cifar10_mean_tensor = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
cifar10_std_tensor = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
