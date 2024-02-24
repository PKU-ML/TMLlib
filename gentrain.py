import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from tqdm import tqdm
import numpy as np
import copy

from utils import Logger, save_checkpoint, torch_accuracy, AverageMeter, trades_loss, mart_loss
from attacks import *

parser = argparse.ArgumentParser(description='Generalist')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--arch', type=str, default="resnet18", help="decide which network to use,choose from smallcnn, resnet18, WRN")
parser.add_argument('--dataset', type=str, default="cifar10", help="dataset")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')

parser.add_argument('--loss_fn', type=str, default="cent", help="loss function")
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--norm', type=str, default='Linf', help='type of attack')

parser.add_argument('--resume',type=bool, default=False, help='whether to resume training')
parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')
parser.add_argument('--ablation', type=str, default='', help='ablation study')


args = parser.parse_args()

# Training settings
args.out_dir = os.path.join(args.out_dir, args.ablation)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

args.num_classes = 10 if args.dataset in ['cifar10', 'mnist', 'svhn'] else 100
weight_decay = 3.5e-3 if args.arch == 'resnet18' else 7e-4
seed = 1

if args.arch == 'resnet18':
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr, args.lr, args.lr/10, args.lr/100])[0]
elif args.arch == 'WRN32':
    args.lr = 0.1
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr, args.lr, args.lr/10, args.lr/20])[0]

adjust_beta = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [1.0, 1.0, 1.0, 0.4])[0]


def train(epoch, model, model_st, teacher_at, teacher_st, teacher_mixed, Attackers, optimizer_ST, optimizer_AT, device, descrip_str):
    teacher_at.model.eval()
    teacher_st.model.eval()
    teacher_mixed.model.eval()

    losses = AverageMeter()
    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()

    pbar = tqdm(train_loader)
    pbar.set_description(descrip_str)
    
    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, target = inputs.to(device), target.to(device)

        # loss, logit = trades_loss(model, inputs, target, optimizer, epoch, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0, distance='l_inf', device='device')
        # loss, logit = mart_loss(model, inputs, target, optimizer, epoch, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0, distance='l_inf', device='device')
        x_adv = Attackers.run_specified('PGD_10', model, inputs, target, return_acc=False)

        # For AT update
        model.train()
        lr = adjust_learning_rate(epoch)
        optimizer_AT.param_groups[0].update(lr=lr)
        optimizer_AT.zero_grad()
        logit = model(x_adv) 
        loss_at = nn.CrossEntropyLoss()(logit, target)
        loss_at.backward()
        optimizer_AT.step()

        teacher_at.update_params(model)
        teacher_at.apply_shadow()

        # For ST update
        model_st.train()
        optimizer_ST.param_groups[0].update(lr=lr)
        optimizer_ST.zero_grad()
        nat_logit = model_st(inputs) 
        loss_st = nn.CrossEntropyLoss()(nat_logit, target)
        loss_st.backward()
        optimizer_ST.step()

        teacher_st.update_params(model_st)
        teacher_st.apply_shadow()

        beta = adjust_beta(epoch)

        teacher_mixed.update_params(teacher_at.model, teacher_st.model, beta=beta)
        teacher_mixed.apply_shadow()
        
        if epoch >= 75 and epoch % 5 == 0:
            model.load_state_dict(teacher_mixed.shadow)
            model_st.load_state_dict(teacher_mixed.shadow)


        losses.update(loss_at.item())
        clean_accuracy.update(torch_accuracy(teacher_st.model(inputs), target, (1,))[0].item())
        adv_accuracy.update(torch_accuracy(teacher_at.model(inputs), target, (1,))[0].item())

        pbar_dic['loss'] = '{:.2f}'.format(losses.mean)
        pbar_dic['Acc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)



def test(model, model_st, Attackers, device):
    model.eval()
    model_st.eval()

    clean_accuracy = AverageMeter()
    adv_accuracy = AverageMeter()   
    ema_clean_accuracy = AverageMeter()
    ema_adv_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Testing')

    for batch_idx, (inputs, target) in enumerate(pbar):
        pbar_dic = OrderedDict()
        
        inputs, target = inputs.to(device), target.to(device)

        acc = Attackers.run_specified('NAT', model, inputs, target, return_acc=True)
        adv_acc = Attackers.run_specified('PGD_20', model, inputs, target, category='Madry', return_acc=True)

        ema_acc = Attackers.run_specified('NAT', model_st, inputs, target, return_acc=True)
        ema_adv_acc = Attackers.run_specified('PGD_20', model_st, inputs, target, category='Madry', return_acc=True)

        clean_accuracy.update(acc[0].item())
        adv_accuracy.update(adv_acc[0].item())        
        ema_clean_accuracy.update(ema_acc[0].item())
        ema_adv_accuracy.update(ema_adv_acc[0].item())

        pbar_dic['cleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['advAcc'] = '{:.2f}'.format(adv_accuracy.mean)        
        pbar_dic['ema_cleanAcc'] = '{:.2f}'.format(ema_clean_accuracy.mean)
        pbar_dic['ema_advAcc'] = '{:.2f}'.format(ema_adv_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return clean_accuracy.mean, adv_accuracy.mean, ema_clean_accuracy.mean, ema_adv_accuracy.mean


def attack(model, Attackers, device):
    model.eval()

    clean_accuracy = AverageMeter()
    pgd20_accuracy = AverageMeter()
    pgd100_accuracy = AverageMeter()
    mim_accuracy = AverageMeter()
    cw_accuracy = AverageMeter()
    APGD_ce_accuracy = AverageMeter()
    APGD_dlr_accuracy = AverageMeter()
    APGD_t_accuracy = AverageMeter()
    FAB_t_accuracy = AverageMeter()
    Square_accuracy = AverageMeter()
    aa_accuracy = AverageMeter()

    pbar = tqdm(test_loader)
    pbar.set_description('Attacking all')

    for batch_idx, (inputs, targets) in enumerate(pbar):
        pbar_dic = OrderedDict()

        inputs, targets = inputs.to(device), targets.to(device)

        acc_dict = Attackers.run_all(model, inputs, targets)

        clean_accuracy.update(acc_dict['NAT'][0].item())
        pgd20_accuracy.update(acc_dict['PGD_20'][0].item())
        pgd100_accuracy.update(acc_dict['PGD_100'][0].item())
        mim_accuracy.update(acc_dict['MIM'][0].item())
        cw_accuracy.update(acc_dict['CW'][0].item())
        APGD_ce_accuracy.update(acc_dict['APGD_ce'][0].item())
        APGD_dlr_accuracy.update(acc_dict['APGD_dlr'][0].item())
        APGD_t_accuracy.update(acc_dict['APGD_t'][0].item())
        FAB_t_accuracy.update(acc_dict['FAB_t'][0].item())
        Square_accuracy.update(acc_dict['Square'][0].item())
        aa_accuracy.update(acc_dict['AA'][0].item())

        pbar_dic['clean'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar_dic['PGD20'] = '{:.2f}'.format(pgd20_accuracy.mean)
        pbar_dic['PGD100'] = '{:.2f}'.format(pgd100_accuracy.mean)
        pbar_dic['MIM'] = '{:.2f}'.format(mim_accuracy.mean)
        pbar_dic['CW'] = '{:.2f}'.format(cw_accuracy.mean)
        pbar_dic['APGD_ce'] = '{:.2f}'.format(APGD_ce_accuracy.mean)
        pbar_dic['APGD_dlr'] = '{:.2f}'.format(APGD_dlr_accuracy.mean)
        pbar_dic['APGD_t'] = '{:.2f}'.format(APGD_t_accuracy.mean)
        pbar_dic['FAB_t'] = '{:.2f}'.format(FAB_t_accuracy.mean)
        pbar_dic['Square'] = '{:.2f}'.format(Square_accuracy.mean)
        pbar_dic['AA'] = '{:.2f}'.format(aa_accuracy.mean)
        pbar.set_postfix(pbar_dic)

    return [clean_accuracy.mean, pgd20_accuracy.mean, pgd100_accuracy.mean, mim_accuracy.mean, cw_accuracy.mean, APGD_ce_accuracy.mean, APGD_dlr_accuracy.mean, APGD_t_accuracy.mean, FAB_t_accuracy.mean, Square_accuracy.mean, aa_accuracy.mean]


def main():
    best_ema_acc_adv = 0
    start_epoch = 1

    if args.arch == "smallcnn":
        model = SmallCNN()
    if args.arch == "resnet18":
        model = ResNet18(num_classes=args.num_classes)
    if args.arch == "preactresnet18":
        model = PreActResNet18(num_classes=args.num_classes)
    if args.arch == "WRN32":
        model = Wide_ResNet_Madry(depth=32, num_classes=args.num_classes, widen_factor=10, dropRate=0.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_at = torch.nn.DataParallel(model)
    model_st = copy.deepcopy(model_at)
    teacher_at = EMA(model_at)
    teacher_st = EMA(model_st)
    teacher_mixed = EMA(model_st)
    # model_at = model_at.to(device)
    Attackers = AttackerPolymer(args.epsilon, args.num_steps, args.step_size, args.num_classes, device)

    logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title='reweight')

    if not args.resume:
        optimizer_ST = optim.SGD(model_st.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        optimizer_AT = optim.SGD(model_at.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
        
        logger_test.set_names(['Epoch', 'Natural', 'PGD20', 'ema_Natural', 'ema_PGD20'])

        for epoch in range(start_epoch, args.epochs+1):
            
            descrip_str = 'Training epoch:{}/{}'.format(epoch, args.epochs)
            # start_time = time.time()
            train(epoch, model_at, model_st, teacher_at, teacher_st, teacher_mixed, Attackers, optimizer_ST, optimizer_AT, device, descrip_str)
            # elapsed = round(time.time() - start_time)
            # elapsed = str(datetime.timedelta(seconds=elapsed))
            # print(elapsed)

            nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc = test(teacher_st.model, teacher_mixed.model, Attackers, device=device)
                 
            logger_test.append([epoch, nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc])

            if ema_nat_acc >= 88 and ema_nat_acc + ema_pgd20_acc > best_ema_acc_adv:
                print('==> Updating the teacher model_at..')
                best_ema_acc_adv = ema_nat_acc + ema_pgd20_acc
                torch.save(teacher_mixed.model.state_dict(), os.path.join(args.out_dir, 'ema_bestpoint.pth.tar'))

        # Save the last checkpoint
        torch.save(model_at.state_dict(), os.path.join(args.out_dir, 'lastpoint.pth.tar'))

    # model_at.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar')))
    teacher_mixed.model.load_state_dict(torch.load(os.path.join(args.out_dir, 'ema_bestpoint.pth.tar')))
    res_list1 = attack(teacher_mixed.model, Attackers, device)
    teacher_mixed.model.load_state_dict(torch.load(os.path.join(args.out_dir, 'lastpoint.pth.tar')))
    res_list2 = attack(teacher_mixed.model, Attackers, device)

    logger_test.set_names(['Epoch', 'clean', 'PGD20', 'PGD100', 'MIM', 'CW', 'APGD_ce', 'APGD_dlr', 'APGD_t', 'FAB_t', 'Square', 'AA'])
    logger_test.append([1000000, res_list1[0], res_list1[1], res_list1[2], res_list1[3], res_list1[4], res_list1[5], res_list1[6], res_list1[7], res_list1[8], res_list1[9], res_list1[10]])
    logger_test.append([1000001, res_list2[0], res_list2[1], res_list2[2], res_list2[3], res_list2[4], res_list2[5], res_list2[6], res_list2[7], res_list2[8], res_list2[9], res_list2[10]])

    logger_test.close()


if __name__ == '__main__':
    main()
