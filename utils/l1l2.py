import torch


def get_l1(l1: float, model: torch.nn.Module):
    robust_loss = 0
    if l1 != 0.0:
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                robust_loss += l1 * param.abs().sum()
    return robust_loss


def get_l2(l2: float, model: torch.nn.Module):
    if l2 != 0.0:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        model_params = [{'params': decay, 'weight_decay': l2},
                        {'params': no_decay, 'weight_decay': 0}]
    else:
        model_params = model.parameters()
    return model_params
