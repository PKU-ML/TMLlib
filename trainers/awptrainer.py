import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from params import AWPParam
from models import get_model
from logging import Logger

from utils.awp import AdvWeightPerturb
from utils.attack import attack_pgd, AttackerPolymer
from utils.const import *
from utils.tens import normalize, mixup_data
from utils.loss import mixup_criterion
from utils.lr_schedule import LRSchedule


class AWPTrainer():

    def awp_model_parameters(self):
        if self.param.l2:
            decay, no_decay = [], []
            for name, param in self.model.named_parameters():
                if 'bn' not in name and 'bias' not in name:
                    decay.append(param)
                else:
                    no_decay.append(param)
            model_params = [{'params': decay, 'weight_decay': self.param.l2},
                            {'params': no_decay, 'weight_decay': 0}]
        else:
            model_params = self.model.parameters()
        return model_params

    def __init__(self, param: AWPParam, train_dataloader: DataLoader, val_dataloader: DataLoader, logger: Logger) -> None:

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param

        self.model = get_model(self.param.model, num_classes=param.num_classes)
        # self.model = nn.DataParallel(self.model).cuda()
        self.opt = torch.optim.SGD(self.awp_model_parameters(), lr=self.param.lr_max, momentum=0.9, weight_decay=5e-4)

        self.proxy = get_model(self.param.model, num_classes=param.num_classes)
        # self.proxy = nn.DataParallel(self.proxy).cuda()
        self.proxy_opt = torch.optim.SGD(self.proxy.parameters(), lr=0.01)

        self.best_perf = -float('inf')
        self.current_perf = -float('inf')
        self.start_epoch = 0

        if self.param.resume:
            saved_dict = torch.load(self.param.save_dir / f"last_ckpt.pth")
            self.model.load_state_dict(saved_dict['model_state_dict'])
            self.opt.load_state_dict(saved_dict['opt_state_dict'])
            self.proxy.load_state_dict(saved_dict['proxy_state_dict'])
            self.proxy_opt.load_state_dict(saved_dict['proxy_opt_state_dict'])
            self.start_epoch = saved_dict['epoch']
            self.best_perf = saved_dict['perf']
            self.logger.info(f'Resuming at epoch {self.param.save_dir / f"last_ckpt.pth"}')
            del saved_dict

        self.epoch = self.start_epoch
        self.criterion = nn.CrossEntropyLoss()
        self.lr_schedule = LRSchedule(self.param)
        self.awp_adversary = AdvWeightPerturb(model=self.model, proxy=self.proxy, proxy_optim=self.proxy_opt, gamma=self.param.awp_gamma)

        self.logger.info('Epoch \t \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

    def train_one_epoch(self):

        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0

        for i, (X, y) in enumerate(self.train_dataloader):
            X, y = X.cuda(), y.cuda()
            lr = self.lr_schedule(self.epoch + (i + 1) / len(self.train_dataloader))
            self.opt.param_groups[0].update(lr=lr)
            self.opt.zero_grad()

            self.model.eval()
            if self.param.mixup:
                X, y_a, y_b, lam = mixup_data(X, y, self.param.mixup_alpha)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))

            if self.param.attack == 'pgd':
                # Random initialization
                if self.param.mixup:
                    delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size, self.param.num_steps,
                                       self.param.restarts, self.param.delta_norm, mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                else:
                    delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size, self.param.num_steps,
                                       self.param.restarts, self.param.delta_norm)
                delta = delta.detach()
            elif self.param.attack == 'fgsm':
                delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size, 1, 1, self.param.delta_norm)
                delta = delta.detach()
            # Standard training
            elif self.param.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            self.model.train()
            # calculate adversarial weight perturbation and perturb it
            if self.epoch >= self.param.awp_warmup:
                # not compatible to mixup currently.
                assert (not self.param.mixup)
                awp = self.awp_adversary.calc_awp(inputs_adv=X_adv,
                                                  targets=y)
                self.awp_adversary.perturb(awp)

            robust_output = self.model(X_adv)
            if self.param.mixup:
                robust_loss = mixup_criterion(self.criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = self.criterion(robust_output, y)

            if self.param.l1:
                for name, param in self.model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += self.param.l1*param.abs().sum()

            robust_loss.backward()
            self.opt.step()

            if self.epoch >= self.param.awp_warmup:
                self.awp_adversary.restore(awp)

            output = self.model(normalize(X))
            if self.param.mixup:
                loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
            else:
                loss = self.criterion(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        self.logger.info('train \t %d \t \t %.4f ' +
                         '\t %.4f \t %.4f \t %.4f \t \t %.4f',
                         self.epoch, lr,
                         train_loss / train_n, train_acc / train_n, train_robust_loss / train_n, train_robust_acc / train_n)

    def val_one_epoch(self):

        val_loss = 0
        val_acc = 0
        val_robust_loss = 0
        val_robust_acc = 0
        val_n = 0

        self.model.eval()
        for i, (X, y) in enumerate(self.val_dataloader):
            X, y = X.cuda(), y.cuda()

            # Random initialization
            if self.param.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size,
                                   self.param.num_steps, self.param.restarts, self.param.delta_norm, early_stop=self.param.eval)
            delta = delta.detach()

            robust_output = self.model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
            robust_loss = self.criterion(robust_output, y)

            output = self.model(normalize(X))
            loss = self.criterion(output, y)

            val_robust_loss += robust_loss.item() * y.size(0)
            val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            val_loss += loss.item() * y.size(0)
            val_acc += (output.max(1)[1] == y).sum().item()
            val_n += y.size(0)

        self.logger.info('val   \t %d \t \t %.4f ' +
                         '\t %.4f \t %.4f \t %.4f \t %.4f',
                         self.epoch, 0,
                         val_loss / val_n, val_acc / val_n, val_robust_loss / val_n, val_robust_acc / val_n)

        self.current_perf = val_robust_acc / val_n

        if self.param.eval:
            return

        saved_dict = {
            'model_state_dict': self.model.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'proxy_state_dict': self.proxy.state_dict(),
            'proxy_opt_state_dict': self.proxy_opt.state_dict(),
            'epoch': self.epoch,
            'perf': self.current_perf,
        }

        # save last
        torch.save(saved_dict, self.param.save_dir / f"last_ckpt.pth")

        # save every iters
        if (self.epoch+1) % self.param.save_freq == 0 or self.epoch+1 == self.param.epochs:
            torch.save(saved_dict, self.param.save_dir / f"epoch_{self.epoch}.pth")

        # save best
        if self.current_perf > self.best_perf:
            self.best_perf = val_robust_loss / val_n
            torch.save(saved_dict, self.param.save_dir / f"best_ckpt.pth")

        del saved_dict

    def run(self):

        if self.param.eval:
            self.val_one_epoch()
            return

        for self.epoch in range(self.start_epoch, self.param.epochs):
            self.train_one_epoch()
            self.val_one_epoch()
