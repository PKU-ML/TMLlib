import torch
import torch.nn as nn
import torch.nn.functional as F


class MARTLoss():

    def __init__(self, beta=6.0) -> None:

        self.beta = float(beta)
        self.kl = nn.KLDivLoss(reduction='none')

    def __call__(self, model, X, X_adv, y):

        logits = model(X)
        logits_adv = model(X_adv)
        adv_probs = F.softmax(logits_adv, dim=1)
        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
        nat_probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        loss_robust = (1.0 / len(y)) * torch.sum(
            torch.sum(self.kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        loss = loss_adv + float(self.beta) * loss_robust

        return loss


def pair_cosine_similarity(x, y=None, eps=1e-8):
    if (y == None):
        n = x.norm(p=2, dim=1, keepdim=True)
        return (x @ x.t()) / (n * n.t()).clamp(min=eps)
    else:
        n1 = x.norm(p=2, dim=1, keepdim=True)
        n2 = y.norm(p=2, dim=1, keepdim=True)
        return (x @ y.t()) / (n1 * n2.t()).clamp(min=eps)


def nt_xent(x, y=None, t=0.5):
    if (y != None):
        x = pair_cosine_similarity(x, y)
    else:
        # print("device of x is {}".format(x.device))
        x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()
