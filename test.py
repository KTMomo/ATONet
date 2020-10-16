import os
import numpy as np
import os
import numpy as np

import torch

import torchvision.transforms as standard_transforms
import utils.transforms as extended_transforms
from datasets import new_CityscapesV2 as cityscapes
from nets.ATONet import ATONet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image


args = {
    'test_batch_size': 4,
    'input_size': (1024, 2048),
    'snapshot': 'epoch_73_loss_0.19220_acc_0.95390_acc-cls_0.77585_mean-iu_0.68189_fwavacc_0.91676_lr_0.0010000000.pth',
    'val_img_sample_rate': 0.05,
    'val_save_to_img_file': True,
}

exp_file = 'test_exp'


def main():
    # epoch = 100
    # info = "ATONet_final3_loss3_5_BN_batch=4_use_ohem=0_bins=8_4_2epoch=100"
    # snapshot = "epoch_98_loss_0.12540_acc_0.95847_acc-cls_0.78683_mean-iu_0.70210_fwavacc_0.92424_lr_0.0000453781.pth"
    # epoch=200
    info = "ATONet_final3_loss3_5_BN_batch=4_use_ohem=False_bins=8_4_2epoch=200"
    snapshot = "epoch_193_loss_0.11953_acc_0.96058_acc-cls_0.79683_mean-iu_0.71272_fwavacc_0.92781_lr_0.0000798490.pth"

    model_save_path = './save_models/cityscapes/{}'.format(info)
    print(model_save_path)

    net = ATONet(classes=19, bins=(8, 4, 2), use_ohem=False).cuda()

    net.load_state_dict(torch.load(os.path.join(model_save_path, snapshot)))

    net.eval()
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()

    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    root = '/titan_data1/caokuntao/data/cityscapes'

    test_set = cityscapes.CityScapes(root, 'fine', 'test', transform=input_transform, target_transform=target_transform)
    test_loader = DataLoader(test_set, batch_size=args['test_batch_size'], num_workers=4, shuffle=False)

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    trainid_to_id = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24, 12: 25, 13: 26,
                     14: 27, 15: 28, 16: 31, 17: 32, 18: 33}

    net.eval()

    gts_all, predictions_all, img_name_all = [], [], []
    with torch.no_grad():
        for vi, data in enumerate(test_loader):
            inputs, img_name = data
            N = inputs.size(0)
            inputs = Variable(inputs).cuda()

            outputs = net(inputs)
            predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

            predictions_all.append(predictions)
            img_name_all.append(img_name)

        print('done')
        predictions_all = np.concatenate(predictions_all)
        img_name_all = np.concatenate(img_name_all)

        to_save_dir = os.path.join(model_save_path, exp_file)
        if not os.path.exists(to_save_dir):
            os.mkdir(to_save_dir)

        for idx, data in enumerate(zip(img_name_all, predictions_all)):
            if data[0] is None:
                continue
            img_name = data[0]
            pred = data[1]
            pred_copy = pred.copy()
            for k, v in trainid_to_id.items():
                pred_copy[pred == k] = v
            pred = Image.fromarray(pred_copy.astype(np.uint8))
            pred.save(os.path.join(to_save_dir, img_name))


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


if __name__ == '__main__':
    main()
