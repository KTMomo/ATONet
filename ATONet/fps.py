import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from nets.modules.switchable_norm import SwitchNorm2d
import time

from ptflops import get_model_complexity_info
#  pip install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
from log.log import logger_count

class CityScapes(data.Dataset):
    def __init__(self, root, quality, mode):
        self.imgs = self._make_dataset(root, quality, mode)

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        self.quality = quality
        self.mode = mode
        self.transform = input_transform
        self.w = 1024
        self.h = 1024
        logger_count.info("size: {}_{}".format(self.w, self.h))

    def __getitem__(self, index):
        img_path, _ = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        # img = img.resize((self.w, self.h), Image.BILINEAR)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def _make_dataset(root, quality, mode):
        assert (quality == 'fine' and mode in ['train', 'val', 'test']) or \
               (quality == 'coarse' and mode in ['train', 'train_extra', 'val'])

        if quality == 'coarse':
            img_dir_name = 'leftImg8bit_trainextra' if mode == 'train_extra' else 'leftImg8bit_trainvaltest'
            mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', mode)
            mask_postfix = '_gtCoarse_labelIds.png'
        else:
            mask_path = os.path.join(root, 'gtFine', mode)
            mask_postfix = '_gtFine_labelIds.png'

        img_path = os.path.join(root, 'leftImg8bit', mode)
        assert os.listdir(img_path) == os.listdir(mask_path)
        items = []
        categories = os.listdir(img_path)
        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            for it in c_items:
                item = (os.path.join(img_path, c, it + '_leftImg8bit.png'),
                        os.path.join(mask_path, c, it + mask_postfix))
                items.append(item)

        return items
