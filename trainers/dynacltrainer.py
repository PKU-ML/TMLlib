import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from logging import Logger
from tqdm import tqdm

from params import DynACLParam
from models.sslmodels import get_model_ssl

from utils.attack import PGD_contrastive
from utils.const import *
from utils.lr_schedule import LRSchedule
from utils.avg import AverageMeter
from utils.mixup import *
from utils.l1l2 import *
from utils.lars import LARS
from utils.loss import nt_xent


class DynACLTrainer():

    def __init__(self, param: DynACLParam, train_dataloader: DataLoader, val_dataloader: DataLoader, logger: Logger) -> None:

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param

        self.model = get_model_ssl(self.param.model, self.param.device, num_classes=param.num_classes, twoLayerProj=self.param.twoLayerProj)
        # self.model = nn.DataParallel(self.model).cuda()
        if self.param.optimizer == 'adam':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.param.lr_max, weight_decay=self.param.weight_decay)
        elif self.param.optimizer == 'lars':
            self.opt = LARS(self.model.parameters(), lr=self.param.lr_max, weight_decay=self.param.weight_decay)
        elif self.param.optimizer == 'sgd':
            self.opt = torch.optim.SGD(self.model.parameters(), lr=self.param.lr_max, momentum=self.param.momentum, weight_decay=self.param.weight_decay)
        else:
            raise ValueError("no defined optimizer")

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

        train_loss = AverageMeter("train_robust_loss")

        pbar = tqdm(self.train_dataloader)
        for i, X in enumerate(pbar):
            d = X.size()
            X = X.view(d[0]*2, d[2], d[3], d[4]).cuda()

            # lr_schedule
            lr = self.lr_schedule(self.epoch + (i + 1) / len(self.train_dataloader))
            self.opt.param_groups[0].update(lr=lr)

            # attack
            X_adv = PGD_contrastive(self.model, X, self.param.epsilon, self.param.step_size, self.param.num_steps)
            features_adv = self.model.train()(X_adv, 'pgd', swap=True)
            features = self.model.train()(X, 'normal', swap=True)
            self.model._momentum_update_encoder_k()

            weight_adv = min(1.0 + (self.epoch // self.param.reload_frequency)
                             * (self.param.reload_frequency / self.param.epochs) * self.param.swap_param, 2)

            loss = (nt_xent(features) * (2 - weight_adv) + nt_xent(features_adv) * weight_adv) / 2

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # log
            train_loss.update(loss.item(), d[0])

            pbar.set_description(f'Epoch {self.epoch + 1}/{self.param.epochs}, Loss: {train_loss.mean:.4f}')
            pbar.update()

    def val_one_epoch(self):
        '''
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
        '''

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

        self.reload()
        for self.epoch in range(self.start_epoch, self.param.epochs):
            if self.epoch % self.param.reload_frequency == 0:
                self.reload()
            self.train_one_epoch()
            self.val_one_epoch()

    def reload(self):
        strength = 1 - self.epoch / self.param.epochs
        self.train_dataloader.dataset.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                96 if self.param.dataset == 'stl10' else 32, scale=(1.0 - 0.9 * strength, 1.0)),
            # No need to decay horizontal flip
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(
                0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength),
            transforms.RandomGrayscale(p=0.2 * strength),
            transforms.ToTensor(),
        ])
        # TODO is this OK?
