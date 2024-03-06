from .resnet_multi_bn import resnet18_momentum as resnet18_momentum_multi_bn
from .resnet_multi_bn import proj_head as proj_head_multi_bn
from .resnet_multi_bn_stl import resnet18_momentum as resnet18_momentum_multi_bn_stl
from .resnet_multi_bn_stl import proj_head as proj_head_multi_bn_stl


def get_model_ssl(model_name, device, num_classes: int = 10, twoLayerProj: bool = False, stl: bool = True):

    bn_names = ['normal', 'pgd']

    # define model
    if model_name == "resnet":
        BACKBONE = resnet18_momentum_multi_bn
        PROJHEAD = proj_head_multi_bn
    elif model_name == "resnet_stl":
        BACKBONE = resnet18_momentum_multi_bn_stl
        PROJHEAD = proj_head_multi_bn_stl
    else:
        raise ValueError("Not suppoted model name")

    model = BACKBONE(pretrained=False, bn_names=bn_names)
    ch = model.encoder_k.fc.in_features
    model.encoder_q.fc = PROJHEAD(ch, bn_names=bn_names, twoLayerProj=twoLayerProj)
    model.encoder_k.fc = PROJHEAD(ch, bn_names=bn_names, twoLayerProj=twoLayerProj)
    model._init_encoder_k()
    model.to(device)

    return model
