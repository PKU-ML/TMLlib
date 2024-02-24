import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalize(ori):
    mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
    std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    return (ori - mu) / std


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, n_classes=1000, transform=None):
        super(VGG16, self).__init__()

        self.layer11 = conv_layer(3, 64, 3, 1)
        self.layer12 = conv_layer(64, 64, 3, 1)
        self.maxPool1 = tnn.MaxPool2d(2, 2)

        self.layer21 = conv_layer(64, 128, 3, 1)
        self.layer22 = conv_layer(128, 128, 3, 1)
        self.maxPool2 = tnn.MaxPool2d(2, 2)

        self.layer31 = conv_layer(128, 256, 3, 1)
        self.layer32 = conv_layer(256, 256, 3, 1)
        self.layer33 = conv_layer(256, 256, 3, 1)
        self.maxPool3 = tnn.MaxPool2d(2, 2)

        self.layer41 = conv_layer(256, 512, 3, 1)
        self.layer42 = conv_layer(512, 512, 3, 1)
        self.layer43 = conv_layer(512, 512, 3, 1)
        self.maxPool4 = tnn.MaxPool2d(2, 2)

        self.max_pooling = tnn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.layer6 = vgg_fc_layer(1 * 1 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 512)

        # Final layer
        self.layer8 = tnn.Linear(512, n_classes)

    def forward(self, x):
        # x = normalize(x)
        out = self.layer11(x)
        out = self.layer12(out)
        out = self.maxPool1(out)

        out = self.layer21(out)
        out = self.layer22(out)
        out = self.maxPool2(out)

        out = self.layer31(out)
        out = self.layer32(out)
        out = self.layer33(out)
        out = self.maxPool3(out)

        out = self.layer41(out)
        out = self.layer42(out)
        out = self.layer43(out)
        out = self.maxPool4(out)

        vgg16_features = self.max_pooling(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out

    def get_mid(self, x, layer=0):
        self.eval()
        # x = normalize(x)

        out = self.layer11(x)
        if layer == 0:
            return out
        out = self.layer12(out)
        if layer == 1:
            return out
        out = self.maxPool1(out)
        if layer == 2:
            return out

        out = self.layer21(out)
        if layer == 3:
            return out
        out = self.layer22(out)
        if layer == 4:
            return out
        out = self.maxPool2(out)
        if layer == 5:
            return out

        out = self.layer31(out)
        if layer == 6:
            return out
        out = self.layer32(out)
        if layer == 7:
            return out
        out = self.layer33(out)
        if layer == 8:
            return out
        out = self.maxPool3(out)
        if layer == 9:
            return out

        out = self.layer41(out)
        if layer == 10:
            return out
        out = self.layer42(out)
        if layer == 11:
            return out
        out = self.layer43(out)
        if layer == 12:
            return out
        out = self.maxPool4(out)
        if layer == 13:
            return out

        vgg16_features = self.max_pooling(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out
