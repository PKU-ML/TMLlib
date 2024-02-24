import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
import numpy as np

from params import GeneralistParam
from models import get_model
from logging import Logger

from utils.attack import attack_pgd, AttackerPolymer
from utils.const import *
from utils.tens import normalize
from utils.lr_schedule import LRSchedule
from utils.ema import EMA


class GeneralistTrainer():

    def adjust_beta(self, t):
        return np.interp([t], [0, self.param.epochs // 3, self.param.epochs * 2 // 3, self.param.epochs],
                         [1.0, 1.0, 1.0, 0.4])[0]

    def __init__(self, param: GeneralistParam, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 logger: Logger) -> None:

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param

        self.model = get_model(self.param.model, num_classes=param.num_classes)
        # self.model = nn.DataParallel(self.model).cuda()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.param.lr_max,
                                   momentum=0.9, weight_decay=self.param.weight_decay)

        self.model_ST = get_model(self.param.model, num_classes=param.num_classes)
        # self.model_ST = nn.DataParallel(self.model_ST).cuda()
        self.opt_ST = torch.optim.SGD(self.model_ST.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

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
        self.criterion_ST = nn.CrossEntropyLoss()
        self.lr_schedule = LRSchedule(self.param)
        self.adjust_beta = lambda t: np.interp([t], [0, self.param.epochs // 3, self.param.epochs * 2 // 3, self.param.epochs], [1.0, 1.0, 1.0, 0.4])[0]

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
            beta = self.adjust_beta(self.epoch + (i + 1) / len(self.train_dataloader))
            self.opt.param_groups[0].update(lr=lr)
            self.opt.zero_grad()

            if self.param.attack == 'pgd':
                delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size, self.param.num_steps,
                                   self.param.restarts, self.param.delta_norm)
                delta = delta.detach()
            elif self.param.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            self.model.train()
            logit = self.model(X_adv)
            loss_at = self.criterion(logit, y)
            loss_at.backward()
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


            self.teacher_mixed.update_params(self.teacher_AT.model, self.teacher_ST.model, beta=beta)
            self.teacher_mixed.apply_shadow()

            if self.epoch >= 75 and self.epoch % 5 == 0:
                self.model.load_state_dict(self.teacher_mixed.shadow)
                self.model_ST.load_state_dict(self.teacher_mixed.shadow)

            train_robust_loss += loss_at.item() * y.size(0)
            train_robust_acc += (logit.max(1)[1] == y).sum().item()
            train_loss += loss_st.item() * y.size(0)
            train_acc += (nat_logit.max(1)[1] == y).sum().item()
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
