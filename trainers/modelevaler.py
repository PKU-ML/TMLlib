import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from logging import Logger
from tqdm import tqdm
from collections import OrderedDict
import json

from params import EvalParam
from models import get_model

from utils.const import *
from utils.avg import AverageMeter
from utils.attack import AttackerPolymer


class ModelEvaler():

    def __init__(self, param: EvalParam, val_dataloader: DataLoader, logger: Logger) -> None:

        self.val_dataloader = val_dataloader
        self.logger = logger
        self.param = param
        self.model = get_model(self.param.model, self.param.device, num_classes=param.num_classes)
        saved_dict = torch.load(self.param.ckpt_file)
        self.model.load_state_dict(saved_dict['model_state_dict'])
        del saved_dict
        self.attacker = AttackerPolymer(self.param.epsilon, self.param.num_steps, self.param.step_size, self.param.num_classes, self.param.device)

    def attack(self):

        attack_list = ['NAT', 'PGD_20', 'PGD_100', 'MIM', 'CW',
                       'APGD_ce', 'APGD_dlr', 'APGD_t', 'FAB_t', 'Square', 'AA',]
        self.model.eval()
        accuracy_dict = {key: AverageMeter(key) for key in attack_list}
        pbar = tqdm(self.val_dataloader)
        pbar.set_description('Attacking all')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            pbar_dic = OrderedDict()
            inputs, targets = inputs.to(self.param.device), targets.to(self.param.device)
            acc_dict = self.attacker.run_all(self.model, inputs, targets)
            for key in attack_list:
                accuracy_dict[key].update(acc_dict[key][0].item(), len(targets))
                pbar_dic[key] = '{:.2f}'.format(accuracy_dict[key].mean)
            pbar.set_postfix(pbar_dic)
        result_dict = {key: accuracy_dict[key].mean for key in attack_list}
        return result_dict

    def run(self):

        result_dict = ModelEvaler.attack(self.model, self.attacker, self.val_dataloader)
        result_json_string = json.dumps(result_dict, indent=4)
        self.logger.info(result_json_string)
        return result_dict
