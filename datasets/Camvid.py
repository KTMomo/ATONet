import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data

num_classes = 11
ignore_label = 255
palette = [128, 128, 128,
           128, 0, 0,
           192, 192, 128,
           128, 64, 128,
           0, 0, 192,
           128, 128, 0,
           192, 128, 128,
           64, 64, 128,
           64, 0, 128,
           0, 128, 192]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


# 720 x 960
class Camvid(data.Dataset):
    def __init__(self, root, mode, transform=None,
                 target_transform=None, crop_size_H=720, crop_size_W=720, val_scale=None):
        self.crop_size_W = crop_size_W
        self.crop_size_H = crop_size_H

        self.imgs = self._make_dataset(root, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.val_scale = val_scale
        self.id_to_trainid = {0: 9, 1: 1, 2: 10, 3: 1, 4: 1, 5: 8, 6: 9,
                              7: 9, 8: 2, 9: 7, 10: 3, 11: 3, 12: 6, 13: 10,
                              14: 8, 15: 4, 16: 9, 17: 3, 18: 4, 19: 4, 20: 6,
                              21: 0, 22: 8, 23: 2, 24: 6, 25: 8, 26: 5, 27: 8,
                              28: 1, 29: 5, 30: ignore_label, 31: 1}

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        if self.mode == 'train':
            img, mask = self._train_transform(img, mask, self.crop_size_W, self.crop_size_H)
            img = self.transform(img)
            mask4 = mask.resize((int(self.crop_size_W / 4), int(self.crop_size_H / 4)), Image.NEAREST)
            mask8 = mask4.resize((int(self.crop_size_W / 8), int(self.crop_size_H / 8)), Image.NEAREST)
            mask16 = mask8.resize((int(self.crop_size_W / 16), int(self.crop_size_H / 16)), Image.NEAREST)
            mask32 = mask16.resize((int(self.crop_size_W / 32), int(self.crop_size_H / 32)), Image.NEAREST)

            mask = self.target_transform(mask)
            mask4 = self.target_transform(mask4)
            mask8 = self.target_transform(mask8)
            mask16 = self.target_transform(mask16)
            mask32 = self.target_transform(mask32)
            return img, [mask, mask4, mask8, mask16, mask32]
        elif self.mode == 'val':
            img, mask = img, mask  # Original size: for single scale or multi-scale verification

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # if self.mode == 'test':
        #     img_name = img_path.split('/')[-1]
        #     img_name = img_name.split('.png')[0]
        #     img_name = img_name + '*.png'
        #     return img, img_name

        return img, mask

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def _make_dataset(root, mode):
        assert (mode in ['train', 'val', 'test'])

        img_path = os.path.join(root, mode, 'img')
        mask_path = os.path.join(root, mode, 'label')
        txt_path = os.path.join(root, "%s.txt"%mode)

        f = open(txt_path, 'r')
        name_list = f.readlines()
        f.close()

        items = []
        for name in name_list:
            name = name.split('.')[0]
            item = (os.path.join(img_path, name + ".png"),
                    os.path.join(mask_path, name + "_P.png"))
            items.append(item)

        if mode == 'train':
            f = open(os.path.join(root, "val.txt"), 'r')
            val_list = f.readlines()
            f.close()
            for name in val_list:
                name = name.split('.')[0]
                item = (os.path.join(os.path.join(root, 'val', 'img'), name + ".png"),
                        os.path.join(os.path.join(root, 'val', 'label'), name + "_P.png"))
                items.append(item)
        return items

    @staticmethod
    def _val_transform(img, mask, crop_size_W, crop_size_H):
        w, h = img.size
        x1 = int((w - crop_size_W) / 2)
        y1 = int((h - crop_size_H) / 2)
        img = img.crop((x1, y1, x1 + crop_size_W, y1 + crop_size_H))
        mask = mask.crop((x1, y1, x1 + crop_size_W, y1 + crop_size_H))

        return img, mask

    @staticmethod
    def _train_transform(img, mask, crop_size_W, crop_size_H):
        # random scale [0.75, 2]
        w, h = img.size  # (960,720)
        # scale = 0.75 + 1.25 * random.random()
        # 测试训练图片尺寸和精度的关系
        scale = 1.0 + 1.0 * random.random()
        oh = int(np.ceil(scale * h))
        ow = int(2 * oh)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # random crop crop_size
        x1 = random.randint(0, ow - crop_size_W)
        y1 = random.randint(0, oh - crop_size_H)
        img = img.crop((x1, y1, x1 + crop_size_W, y1 + crop_size_H))
        mask = mask.crop((x1, y1, x1 + crop_size_W, y1 + crop_size_H))
        # random flip left_right
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask
