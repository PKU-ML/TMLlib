import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from logging import Logger
import copy

from params import ReBATParam
from models import get_model

from utils.attack import attack_pgd
from utils.const import *
from utils.tens import normalize
from utils.lr_schedule import LRSchedule
from utils.avg import AverageMeter
from utils.mixup import *
from utils.l1l2 import *
from utils.attack import AttackerPolymer
from utils.misc import moving_average

# TODO support torch.utils.tensorboard.writer.SummaryWriter


class ReBATTrainer():

    def __init__(self, param: ReBATParam, train_dataloader: DataLoader, val_dataloader: DataLoader, logger: Logger) -> None:

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param

        self.model = get_model(self.param.model, num_classes=param.num_classes)
        # self.model = nn.DataParallel(self.model).cuda()
        self.opt = torch.optim.SGD(get_l2(self.param.l2, self.model), lr=self.param.lr_max,
                                   momentum=self.param.momentum, weight_decay=self.param.weight_decay)

        self.model_wa = copy.deepcopy(self.model)

        self.best_perf = -float('inf')
        self.current_perf = -float('inf')
        self.start_epoch = 0

        if self.param.resume:
            saved_dict = torch.load(self.param.save_dir / f"last_ckpt.pth")
            self.model.load_state_dict(saved_dict['model_state_dict'])
            self.opt.load_state_dict(saved_dict['opt_state_dict'])
            self.model_wa.load_state_dict(saved_dict['model_wa_state_dict'])
            self.start_epoch = saved_dict['epoch']
            self.best_perf = saved_dict['perf']
            self.logger.info(f'Resuming at epoch {self.param.save_dir / f"last_ckpt.pth"}')
            del saved_dict

        self.epoch = self.start_epoch
        self.criterion = nn.CrossEntropyLoss()
        self.lr_schedule = LRSchedule(param=self.param)
        self.reg_scheduler = self.get_reg_schedule()
        self.attacker = AttackerPolymer(self.param.epsilon, self.param.num_steps, self.param.step_size, self.param.num_classes, device)

        self.logger.info('Epoch \t \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')

    def train_one_epoch(self):

        train_robust_loss = AverageMeter("train_robust_loss")
        train_robust_acc = AverageMeter("train_robust_acc")
        train_reg_loss = AverageMeter("train_reg_loss")

        decay_rate = self.param.decay_rate if self.epoch >= self.param.warmup_epochs else 0.  # for WA
        boat_beta = self.param.boat_beta if self.epoch >= self.param.warmup_epochs else 0.  # force deactivating BoAT regularization before WA starts

        for i, (X, y) in enumerate(self.train_dataloader):
            X, y = X.cuda(), y.cuda()
            if self.param.cutmix:
                X, y_a, y_b, lam = cutmix_data(X, y, self.param.cutmix_alpha, self.param.cutmix_beta)
                X, y_a, y_b = map(Variable, (X, y_a, y_b))

            # lr_schedule
            lr = self.lr_schedule(self.epoch + (i + 1) / len(self.train_dataloader))  # TODO to be fixed
            self.opt.param_groups[0].update(lr=lr)
            self.opt.zero_grad()

            # attack
            self.model.eval()
            if self.param.attack == 'pgd':
                if not self.param.stronger_attack or self.lr_schedule.stage(self.epoch) < 1:  # ReBAT[strong] # TODO
                    if self.param.cutmix:
                        delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size,
                                           self.param.num_steps, self.param.restarts, self.param.delta_norm,
                                           mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(self.model, X, y, self.param.epsilon, self.param.step_size,
                                           self.param.num_steps, self.param.restarts, self.param.delta_norm)
                else:
                    if self.param.cutmix:
                        delta = attack_pgd(self.model, X, y, self.param.stronger_epsilon, self.param.step_size,
                                           self.param.stronger_num_steps, self.param.restarts, self.param.delta_norm,
                                           mixup=True, y_a=y_a, y_b=y_b, lam=lam)
                    else:
                        delta = attack_pgd(self.model, X, y, self.param.stronger_epsilon, self.param.step_size,
                                           self.param.stronger_num_steps, self.param.restarts, self.param.delta_norm)
            elif self.param.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                ValueError("Error attack type")
            delta = delta.detach()
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            # train
            self.model.train()
            robust_output = self.model(X_adv)
            if self.param.cutmix:
                robust_loss = mixup_criterion(self.criterion, robust_output, y_a, y_b, lam)
            else:
                robust_loss = self.criterion(robust_output, y)

            if boat_beta > 0:  # apply BoAT loss
                reg_loss = torch.tensor(0.).cuda()
                with torch.no_grad():
                    robust_output_wa = self.model_wa(X_adv)
                reg_loss = F.kl_div(F.log_softmax(robust_output, dim=1),
                                    F.softmax(robust_output_wa, dim=1), reduction='batchmean')
                if reg_loss < 1e10:
                    if self.param.use_reg_schedule:
                        boat_beta = self.reg_scheduler(self.epoch + (i + 1) / len(self.train_dataloader))
                    robust_loss += reg_loss * boat_beta

            robust_loss += get_l1(self.param.l1, self.model)
            robust_loss.backward()
            self.opt.step()

            moving_average(self.model_wa, self.model, decay_rate, update_bn=True)

            # log
            train_robust_loss.update(robust_loss.item(), len(y))
            train_robust_acc .update((robust_output.max(1)[1] == y).mean().item(), len(y))
            train_reg_loss.update(reg_loss.item(), len(y))

        self.logger.info('train \t %d \t \t %.4f \t %.4f \t %.4f \t %.4f',
                         self.epoch, lr, train_robust_loss.mean, train_robust_acc.mean, train_reg_loss.mean)

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
                if self.param.stronger_attack and self.param.stronger_eval:
                    delta = attack_pgd(self.model, X, y, self.param.stronger_epsilon, self.param.step_size,
                                       self.param.stronger_num_steps, self.param.restarts, self.param.delta_norm)
                else:
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
            val_robust_acc.update((robust_output.max(1)[1] == y).mean().item(), len(y))
            val_loss.update(loss.item(), len(y))
            val_acc.update((output.max(1)[1] == y).mean().item(), len(y))

        self.logger.info('val   \t %d \t \t %.4f ' +
                         '\t %.4f \t %.4f \t %.4f \t %.4f',
                         self.epoch, 0,
                         val_loss.mean, val_acc.mean, val_robust_loss.mean, val_robust_acc.mean)

        self.current_perf = val_robust_acc.mean

        saved_dict = {
            'model_state_dict': self.model.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'model_wa_state_dict': self.model_wa.state_dict(),
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

    def get_reg_schedule(self):  # TODO to be fixed
        if self.param.reg_schedule == 'piecewise':
            def reg_schedule(t):
                if self.lr_schedule.stage(t) < 2:  # WA and BoAT regularization start after the first LR decay, usually at epoch 105
                    return self.param.boat_beta
                else:
                    return self.param.boat_beta * self.param.boat_beta_factor
        elif self.param.reg_schedule == 'dependent':
            def reg_schedule(t):
                rate = self.lr_schedule(t)
                return (self.param.lr_max / rate - 1) / 2
        else:
            raise NotImplementedError("Unknown regularization schedule!")
        return reg_schedule
