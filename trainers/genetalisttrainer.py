import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from logging import Logger
from tqdm import tqdm
import copy

from params import GeneralistParam
from models import get_model

from utils.attack import attack_pgd
from utils.const import *
from utils.tens import normalize
from utils.lr_schedule import LRSchedule
from utils.avg import AverageMeter
from utils.mixup import *
from utils.l1l2 import *
from utils.ema import EMA
from utils.avg import AverageMeter


class GeneralistTrainer():

    def __init__(self, param: GeneralistParam, train_dataloader: DataLoader, val_dataloader: DataLoader, logger: Logger) -> None:

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param

        self.model = get_model(self.param.model, self.param.device, num_classes=param.num_classes)
        # self.model = nn.DataParallel(self.model).cuda()
        self.opt = torch.optim.SGD(get_l2(self.param.l2, self.model), lr=self.param.lr_max,
                                   momentum=self.param.momentum, weight_decay=self.param.weight_decay)

        self.model_ST = get_model(self.param.model, self.param.device, num_classes=param.num_classes)
        # self.model_ST = nn.DataParallel(self.model_ST).cuda()
        self.opt_ST = torch.optim.SGD(self.model_ST.parameters(), lr=0.01,
                                      momentum=0.9, weight_decay=5e-4)  # TODO  add parameters about opt_ST

        self.best_perf = -float('inf')
        self.current_perf = -float('inf')
        self.start_epoch = 0

        if self.param.resume:
            saved_dict = torch.load(self.param.save_dir / f"last_ckpt.pth")
            self.model.load_state_dict(saved_dict['model_state_dict'])
            self.opt.load_state_dict(saved_dict['opt_state_dict'])
            self.model_ST.load_state_dict(saved_dict['model_ST_state_dict'])
            self.opt_ST.load_state_dict(saved_dict['opt_ST_state_dict'])
            self.start_epoch = saved_dict['epoch']
            self.best_perf = saved_dict['perf']
            self.logger.info(f'Resuming at epoch {self.param.save_dir / f"last_ckpt.pth"}')
            del saved_dict

        self.teacher_ST = EMA(self.model_ST)
        self.teacher_AT = EMA(self.model)
        self.teacher_mixed = EMA(self.model_ST)

        self.epoch = self.start_epoch
        self.criterion = nn.CrossEntropyLoss()
        self.lr_schedule = LRSchedule(param=self.param)
        self.criterion_ST = nn.CrossEntropyLoss()
        self.beta_schedule = lambda t: np.interp(t, [0, self.param.epochs // 3, self.param.epochs * 2 // 3, self.param.epochs], [1.0, 1.0, 1.0, 0.4])

    def train_one_epoch(self):

        train_loss = AverageMeter("train_loss")
        train_acc = AverageMeter("train_acc")
        train_robust_loss = AverageMeter("train_robust_loss")
        train_robust_acc = AverageMeter("train_robust_acc")

        pbar = tqdm(self.train_dataloader)
        for i, (X, y) in enumerate(pbar):
            X, y = X.cuda(), y.cuda()
            lr = self.lr_schedule(self.epoch + (i + 1) / len(self.train_dataloader))
            self.opt.param_groups[0].update(lr=lr)
            self.opt.zero_grad()

            # attack
            self.model.eval()
            if self.param.attack == 'pgd':
                delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size,
                                   self.param.num_steps, self.param.restarts, self.param.delta_norm)
            elif self.param.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                ValueError("Error attack type")
            delta = delta.detach()
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            # train
            self.model.train()
            robust_output = self.model(X_adv)
            robust_loss = self.criterion(robust_output, y)
            robust_loss += get_l1(self.param.l1, self.model)
            robust_loss.backward()
            self.opt.step()

            self.teacher_AT.update_params(self.model)
            self.teacher_AT.apply_shadow()

            # For ST update
            self.model_ST.train()
            self.opt_ST.param_groups[0].update(lr=lr)
            self.opt_ST.zero_grad()
            nat_logit = self.model_ST(X)
            loss_st = self.criterion_ST(nat_logit, y)
            loss_st.backward()
            self.opt_ST.step()

            self.teacher_ST.update_params(self.model_ST)
            self.teacher_ST.apply_shadow()

            beta = self.beta_schedule(self.epoch + (i + 1) / len(self.train_dataloader))
            self.teacher_mixed.update_params(self.teacher_AT.model, self.teacher_ST.model, beta=beta)
            self.teacher_mixed.apply_shadow()

            if self.epoch >= 75 and self.epoch % 5 == 0:
                self.model.load_state_dict(self.teacher_mixed.shadow)
                self.model_ST.load_state_dict(self.teacher_mixed.shadow)

            # log
            train_robust_loss.update(robust_loss.item(), len(y))
            train_robust_acc.update((robust_output.max(1)[1] == y).sum().item() / len(y), len(y))
            train_loss.update(loss_st.item(), len(y))
            train_acc.update((nat_logit.max(1)[1] == y).sum().item() / len(y), len(y))

            pbar.set_description(f'Epoch {self.epoch + 1}/{self.param.epochs}, Loss: {train_robust_loss.mean:.4f}|{train_loss.mean:.4f}')
            pbar.update()

    def val_one_epoch(self):

        val_loss = AverageMeter("val_loss")
        val_acc = AverageMeter("val_acc")
        val_robust_loss = AverageMeter("val_robust_loss")
        val_robust_acc = AverageMeter("val_robust_acc")

        self.model.eval()
        for i, (X, y) in enumerate(self.val_dataloader):
            X, y = X.cuda(), y.cuda()

            # attack
            if self.param.attack == 'pgd':
                delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size,
                                   self.param.num_steps, self.param.restarts, self.param.delta_norm)
            elif self.param.attack == 'none':
                delta = torch.zeros_like(X)
            delta = delta.detach()
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            # eval
            robust_output = self.model(X_adv)
            robust_loss = self.criterion(robust_output, y)
            output = self.model(normalize(X))
            loss = self.criterion(output, y)

            # log
            val_robust_loss.update(robust_loss.item(), len(y))
            val_robust_acc.update((robust_output.max(1)[1] == y).sum().item() / len(y), len(y))
            val_loss.update(loss.item(), len(y))
            val_acc.update((output.max(1)[1] == y).sum().item() / len(y), len(y))

        self.logger.info('val   \t %d \t \t %.4f ' +
                         '\t %.4f \t %.4f \t %.4f \t %.4f',
                         self.epoch, 0,
                         val_loss.mean, val_acc.mean, val_robust_loss.mean, val_robust_acc.mean)

        self.current_perf = val_robust_acc.mean

        saved_dict = {
            'model_state_dict': self.model.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'model_ST_state_dict': self.model_ST.state_dict(),
            'opt_ST_state_dict': self.opt_ST.state_dict(),
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
            self.best_perf = self.current_perf
            torch.save(saved_dict, self.param.save_dir / f"best_ckpt.pth")

        del saved_dict

    def run(self):

        for self.epoch in range(self.start_epoch, self.param.epochs):
            self.train_one_epoch()
            self.val_one_epoch()
