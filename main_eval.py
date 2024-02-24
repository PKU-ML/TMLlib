from utils.avg import AverageMeter
from tqdm import tqdm
from collections import OrderedDict

def attack(model, Attackers, val_dataloader, device):
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

    pbar = tqdm(val_dataloader)
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
