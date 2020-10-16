import os
import numpy as np
import time
import torchvision.transforms as standard_transforms
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.tools import evaluate, check_makedirs
import utils.transforms as extended_transforms

from datasets import new_CityscapesV2 as cityscapes
from datasets import Camvid as camvid

from fps import CityScapes as val_cityscapes

from nets.ATONet import ATONet
from nets.modules.switchable_norm import SwitchNorm2d
# from ptflops import get_model_complexity_info
from log.log import logger, logger_count

args = {}


def main(network, train_batch_size=4, val_batch_size=4, epoch_num=50,
         lr=2e-2, weight_decay=1e-4, momentum=0.9, factor=10, val_scale=None,
         model_save_path='./save_models/cityscapes', data_type='Cityscapes', snapshot='', accumulation_steps=1):

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    # Loading dataset
    if data_type == 'Cityscapes':
        # dataset_path = '/home/caokuntao/data/cityscapes'
        # dataset_path = '/titan_data2/ckt/datasets/cityscapes'  # 23341
        dataset_path = '/titan_data1/caokuntao/data/cityscapes'  # 新的23341
        train_set = cityscapes.CityScapes(dataset_path, 'fine', 'train', transform=input_transform, target_transform=target_transform)
        val_set = cityscapes.CityScapes(dataset_path, 'fine', 'val', val_scale=val_scale, transform=input_transform, target_transform=target_transform)
    else:
        dataset_path = '/home/caokuntao/data/camvid'
        # dataset_path = '/titan_data1/caokuntao/data/camvid'  # 新的23341
        train_set = camvid.Camvid(dataset_path, 'train', transform=input_transform, target_transform=target_transform)
        val_set = camvid.Camvid(dataset_path, 'test', val_scale=val_scale, transform=input_transform, target_transform=target_transform)

    train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, num_workers=val_batch_size, shuffle=False)

    if len(snapshot) == 0:
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        logger.info('training resumes from ' + snapshot)
        network.load_state_dict(torch.load(os.path.join(model_save_path, snapshot)))
        split_snapshot = snapshot.split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                               'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cityscapes.ignore_label)

    paras = dict(network.named_parameters())
    paras_new = []
    for k, v in paras.items():
        if 'layer' in k and ('conv' in k or 'downsample.0' in k):
            paras_new += [{'params': [v], 'lr': lr / factor, 'weight_decay': weight_decay / factor}]
        else:
            paras_new += [{'params': [v], 'lr': lr, 'weight_decay': weight_decay}]

    optimizer = torch.optim.SGD(paras_new, momentum=momentum)
    lr_sheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_num, eta_min=1e-6)

    # if len(snapshot) > 0:
    #     optimizer.load_state_dict(torch.load(os.path.join(model_save_path, 'opt_' + snapshot)))

    check_makedirs(model_save_path)

    all_iter = epoch_num * len(train_loader)

    #
    # validate(val_loader, network, criterion, optimizer, curr_epoch, restore_transform, model_save_path)
    # return

    for epoch in range(curr_epoch, epoch_num + 1):
        train(train_loader, network, optimizer, epoch, all_iter, accumulation_steps)
        validate(val_loader, network, criterion, optimizer, epoch, restore_transform, model_save_path)
        lr_sheduler.step()

    # 1024 x 2048
    # dataset_path = '/titan_data1/caokuntao/data/cityscapes'  # 新的23341
    # val_set = cityscapes.CityScapes(dataset_path, 'fine', 'val', val_scale=True, transform=input_transform,
    #                                 target_transform=target_transform)
    # val_loader = DataLoader(val_set, batch_size=val_batch_size, num_workers=val_batch_size, shuffle=False)
    # validate(val_loader, network, criterion, optimizer, epoch, restore_transform, model_save_path)

    return
    # # cityscapes
    # val_set = val_cityscapes(dataset_path, 'fine', 'val')
    # val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)
    n = len(val_loader)
    device = torch.device('cuda')
    net.eval()
    with torch.no_grad():
        # torch.cuda.synchronize()
        time_all = 0
        for vi, inputs in enumerate(val_loader):
            inputs = inputs[0].to(device)
            t0 = 1000 * time.time()
            outputs = net(inputs)
            torch.cuda.synchronize()
            t1 = 1000 * time.time()
            time_all = time_all + t1 - t0
            # predictions = outputs.data.max(1)[1].squeeze_(1).cpu()
            # torch.cuda.synchronize()

        fps = (1000 * n) / time_all
        # 每秒多少张
        print(fps)

    #     logger_count.info("fps={}".format(fps))
    # logger_count.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    #

def train(train_loader, net, optimizer, epoch, all_iter, accumulation_steps):
    net.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        assert inputs.size()[2:] == targets[0].size()[1:]
        inputs = Variable(inputs).cuda()
        targets0 = Variable(targets[0]).cuda()
        targets1 = Variable(targets[1]).cuda()
        targets2 = Variable(targets[2]).cuda()
        targets3 = Variable(targets[3]).cuda()
        targets4 = Variable(targets[4]).cuda()

        loss = net(inputs, [targets0, targets1, targets2, targets3, targets4])

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 20 == 0:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info('[%s], [epoch %d], [iter %d / %d], [train loss %.5f]' % (
                localtime, epoch, i + 1, len(train_loader), train_loss / (i + 1)
            ))


def validate(val_loader, net, criterion, optimizer, epoch, restore, model_save_path):
    net.eval()
    val_loss = 0.0
    gts_all, predictions_all = [], []
    count = len(val_loader)
    with torch.no_grad():
        for vi, (inputs, targets) in enumerate(val_loader):
            inputs = Variable(inputs).cuda()
            targets = Variable(targets).cuda()
            outputs = net(inputs)

            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            val_loss += criterion(outputs, targets).item()

            gts_all.append(targets.data.cpu().numpy())
            predictions_all.append(predictions)

        gts_all = np.concatenate(gts_all)
        predictions_all = np.concatenate(predictions_all)

        acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, cityscapes.num_classes)

        if mean_iu > args['best_record']['mean_iu']:
            args['best_record']['val_loss'] = val_loss / count
            args['best_record']['epoch'] = epoch
            args['best_record']['acc'] = acc
            args['best_record']['acc_cls'] = acc_cls
            args['best_record']['mean_iu'] = mean_iu
            args['best_record']['fwavacc'] = fwavacc
            snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
                epoch, val_loss / count, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[0]['lr']
            )

            torch.save(net.state_dict(), os.path.join(model_save_path, snapshot_name + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(model_save_path, 'opt_' + snapshot_name + '.pth'))

        logger.info(
            '-----------------------------------------------------------------------------------------------------------')
        logger.info('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [lr %.10f]' % (
            epoch, val_loss / count, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[0]['lr']))

        logger.info('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
            args['best_record']['val_loss'], args['best_record']['acc'], args['best_record']['acc_cls'],
            args['best_record']['mean_iu'], args['best_record']['fwavacc'], args['best_record']['epoch']))

        logger.info('-----------------------------------------------------------------------------------------------')
        logger_count.info('------------------------------------------------------------------------------------------')
        logger_count.info(
            '[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [lr %.10f]' % (
                epoch, val_loss / count, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[0]['lr']))

        logger_count.info(
            'best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
                args['best_record']['val_loss'], args['best_record']['acc'], args['best_record']['acc_cls'],
                args['best_record']['mean_iu'], args['best_record']['fwavacc'], args['best_record']['epoch']))

        logger_count.info('------------------------------------------------------------------------------------------')

    return val_loss


if __name__ == '__main__':
    # main(network, train_batch_size=8, val_batch_size=8, epoch_num=50,
    #          lr=1e-2, weight_decay=1e-4, momentum=0.9, factor=4,
    #          model_save_path='./save_models/cityscapes', dataset_path='/home/caokuntao/data/cityscapes',
    #          snapshot='', accumulation_steps=1):

    net_name = "camvid720x720_ATONet_final3_loss3_5_"
    norm = 'BN_'
    bins = (8, 4, 2)
    use_ohem = False
    batch = 16
    epoch = 400
    info = net_name + norm + "batch={}_".format(str(batch)) \
           + "use_ohem={}_".format(str(use_ohem)) + "bins={}".format('_'.join(list(map(str, bins)))) \
           + "epoch={}".format(str(epoch))
    logger.info(info)
    snapshot = ""

    net = ATONet(classes=11, bins=bins, use_ohem=use_ohem).cuda()
    # print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    # macs, params = get_model_complexity_info(net, (3, 1024, 2048), as_strings=True, print_per_layer_stat=True,
    #                                          verbose=True)
    # print('{:<30} {:<8}'.format('Comutational complexity: ', macs))
    # print('{:<30} {:<8}'.format('Params: ', params))
    main(net, train_batch_size=batch, lr=2e-2, epoch_num=epoch, val_scale=None,
         model_save_path='./save_models/cityscapes/{}'.format(info), data_type='camvid',
         snapshot=snapshot)
    # 自己的方法：也可以搞成在线的模式，以难样本类别（设定一定比例）反向传播
    #/home/caokuntao/rtSeg/save_models/cityscapes/spp_val1024_10x2e-2_epoch100_bins_8_4_2_1/epoch_97_loss_0.16706_acc_0.94624_acc-cls_0.78459_mean-iu_0.68817_fwavacc_0.90282_lr_0.0000798490.pth


"""
iou: [0.9763195  0.81512812 0.91473005 0.56290567 0.55169627 0.58968516
 0.62645517 0.7316448  0.91730691 0.62033781 0.94485547 0.7787342
 0.54678132 0.93779866 0.74821063 0.83062105 0.68242554 0.5910881
 0.73281822]
2020-07-18 11:07:10,808 - -----------------------------------------------------------------------------------------------------------
2020-07-18 11:07:10,809 - [epoch 99], [val loss 0.13957], [acc 0.95406], [acc_cls 0.82710], [mean_iu 0.74208], [fwavacc 0.91570], [lr 0.0000453781]
2020-07-18 11:07:10,809 - best record: [val loss 0.13957], [acc 0.95406], [acc_cls 0.82710], [mean_iu 0.74208], [fwavacc 0.91570], [epoch 99]
______________________________________
iou: [0.98392685 0.75082507 0.90692477 0.45043951 0.40666313 0.5179302
 0.59208674 0.67354209 0.9252624  0.53733502 0.94921588 0.77015265
 0.53183428 0.915186   0.7209759  0.80766206 0.70921812 0.5115728
 0.67911336]
2020-07-17 13:46:12,425 - -----------------------------------------------------------------------------------------------------------
2020-07-17 13:46:12,425 - [epoch 98], [val loss 0.12540], [acc 0.95847], [acc_cls 0.78683], [mean_iu 0.70210], [fwavacc 0.92424], [lr 0.0000453781]
2020-07-17 13:46:12,425 - best record: [val loss 0.12540], [acc 0.95847], [acc_cls 0.78683], [mean_iu 0.70210], [fwavacc 0.92424], [epoch 98]
"""