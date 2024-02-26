from .mobilenetv2 import MobileNetV2
from .preactresnet import PreActResNet, PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from .resnet import ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .small_cnn import SmallCNN, small_cnn
from .vgg import VGG16
from .wideresnet import WideResNet
from .wideresnetmadry import WideResNetMadry


def get_model(model_name, device, num_classes=10):

    if model_name == "resnet":
        return ResNet18(num_classes=num_classes).to(device)
    elif model_name == "preactresnet":
        return PreActResNet18(num_classes=num_classes).to(device)
    elif model_name == "smallcnn":
        return small_cnn(num_classes=num_classes).to(device)
    elif model_name == "vgg":
        return VGG16(n_classes=num_classes).to(device)
    elif model_name == "mobilenet":
        return MobileNetV2(num_classes=num_classes).to(device)
    elif model_name == "wideresnet":
        return WideResNet(34, num_classes).to(device)
    elif model_name == "wideresnetmadry":
        return WideResNetMadry(34, num_classes).to(device)
    else:
        raise ValueError("Unknown model")
