import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from logging import Logger
from tqdm import tqdm

from params import MARTParam
from models import get_model

from utils.attack import attack_pgd
from utils.const import *
from utils.tens import normalize
from utils.lr_schedule import LRSchedule
from utils.avg import AverageMeter
from utils.mixup import *
from utils.l1l2 import *
from utils.loss import mart_loss


class MARTTrainer():

    def __init__(self, param: MARTParam, train_dataloader: DataLoader, val_dataloader: DataLoader, logger: Logger) -> None:

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param

        self.model = get_model(self.param.model, self.param.device, num_classes=param.num_classes)
        # self.model = nn.DataParallel(self.model).cuda()
        self.opt = torch.optim.SGD(get_l2(self.param.l2, self.model), lr=self.param.lr_max,
                                   momentum=self.param.momentum, weight_decay=self.param.weight_decay)

        self.best_perf = -float('inf')
        self.current_perf = -float('inf')
        self.start_epoch = 0

        if self.param.resume:
            saved_dict = torch.load(self.param.save_dir / f"last_ckpt.pth")
            self.model.load_state_dict(saved_dict['model_state_dict'])
            self.opt.load_state_dict(saved_dict['opt_state_dict'])
            self.start_epoch = saved_dict['epoch']
            self.best_perf = saved_dict['perf']
            self.logger.info(f'Resuming at epoch {self.param.save_dir / f"last_ckpt.pth"}')
            del saved_dict

        self.epoch = self.start_epoch
        self.criterion = nn.CrossEntropyLoss()
        self.lr_schedule = LRSchedule(param=self.param)

    def train_one_epoch(self):

        # TODO BUG: Output NaN result
        train_loss = AverageMeter("train_loss")

        self.model.train()
        pbar = tqdm(self.train_dataloader)
        for i, (X, y) in enumerate(pbar):
            X, y = X.cuda(), y.cuda()
            lr = self.lr_schedule(self.epoch + (i + 1) / len(self.train_dataloader))
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
            # self.opt.param_groups[0].update(lr=lr)
            self.opt.zero_grad()

            # calculate robust loss
            loss = mart_loss(model=self.model,
                             x_natural=X,
                             y=y,
                             optimizer=self.opt,
                             step_size=self.param.step_size,
                             epsilon=self.param.epsilon,
                             perturb_steps=self.param.num_steps,
                             beta=self.param.mart_beta)

            loss.backward()
            self.opt.step()

            # log
            train_loss.update(loss.item(), len(y))

            pbar.set_description(f'Epoch {self.epoch + 1}/{self.param.epochs}, Loss: {loss.item():.4f}')
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
