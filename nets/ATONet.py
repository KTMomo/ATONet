# import torchvision.models as models
import nets.c_resnet as models
import torch.nn.functional as F

from nets.modules.switchable_norm import SwitchNorm2d
import torch
from torch import nn
from nets.modules.loss import OhemCrossEntropy2d

# origin unet + spp, no late

class ATONet(nn.Module):
    def __init__(self, classes=19, bins=(8, 4, 2), norm=nn.BatchNorm2d, use_ohem=False):
        super(ATONet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        if use_ohem:
            self.criterion = OhemCrossEntropy2d(ignore_label=255)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.up1 = singleUp(64, 128, 1)
        self.up2 = singleUp(128, 128, 2)
        self.up3 = singleUp(256, 128, 4)
        self.up4 = singleUp(512, 128, 8)
        # spp
        self.spp = SPP(512, 128, bins)
        self.up5 = singleUp(128, 128, 8)

        # no spp
        # self.up4 = singleUp(512, 128, 8)

        self.relu = nn.ReLU(inplace=True)

        #
        self.final_ocnv = conv(128, 128, 3, 1)
        # 23341 aux_loss,first
        # self.final_ocnv = conv(128, 128, 1)

        self.aux_cls1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classes, kernel_size=1)
        )
        self.aux_cls2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, classes, kernel_size=1)
        )
        self.aux_cls3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, classes, kernel_size=1)
        )
        self.aux_cls4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, classes, kernel_size=1)
        )
        self.aux_cls5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, classes, kernel_size=1)
        )

        self.signal = [False, False, True, True, True]  # aux_loss use

        # cls
        # 23321 128--》 64
        self.cls = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, classes, kernel_size=1)
        )

    def forward(self, x, y=None):
        imsize = x.size()[2:]
        imsize4 = [int(imsize[0]//4), int(imsize[1]//4)]

        x = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.spp(c4)  # 128

        if self.training and self.signal[0]:
            temp = self.aux_cls1(c1)
            loss1 = self.criterion(temp, y[1])
        c1 = self.up1(c1)

        if self.training and self.signal[1]:
            temp = self.aux_cls2(c2)
            loss2 = self.criterion(temp, y[2])
        c2 = self.up2(c2)
        c2 = F.interpolate(c2, scale_factor=2, mode='bilinear', align_corners=False)

        if self.training and self.signal[2]:
            temp = self.aux_cls3(c3)
            loss3 = self.criterion(temp, y[3])
        c3 = self.up3(c3)  # 128
        c3 = F.interpolate(c3, scale_factor=4, mode='bilinear', align_corners=False)

        if self.training and self.signal[3]:
            temp = self.aux_cls4(c4)
            # camvid
            temp = F.interpolate(temp, size=[int(imsize4[0]//8), int(imsize4[1]//8)], mode='bilinear', align_corners=False)

            loss4 = self.criterion(temp, y[4])
        c4 = self.up4(c4)
        c4 = F.interpolate(c4, size=imsize4, mode='bilinear', align_corners=False)

        if self.training and self.signal[4]:
            temp = self.aux_cls5(c5)
            # camvid
            temp = F.interpolate(temp, size=[int(imsize4[0] // 8), int(imsize4[1] // 8)], mode='bilinear',
                                 align_corners=False)
            loss5 = self.criterion(temp, y[4])
        c5 = self.up5(c5)
        c5 = F.interpolate(c5, size=imsize4, mode='bilinear', align_corners=False)
        # 计算辅助损失

        out = c1 + c2 + c3 + c4 + c5
        out = self.relu(out)
        out = self.final_ocnv(out)

        out = self.cls(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear')

        if self.training:
            main_loss = self.criterion(out, y[0])
            if self.signal[0]:
                main_loss += loss1
            if self.signal[1]:
                main_loss += loss2
            if self.signal[2]:
                main_loss += loss3
            if self.signal[3]:
                main_loss += loss4
            if self.signal[4]:
                main_loss += loss5
            return main_loss
        else:
            # out = F.interpolate(out, scale_factor=2, mode='bilinear')
            return out


def conv(in_planes, out_planes, k, padding=0, norm=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k, padding=padding, bias=False),
        norm(out_planes),
        nn.ReLU(inplace=True))


class SPP(nn.Module):
    def __init__(self, in_dim=512, reduction_dim=128, bins=(6, 3, 2, 1), norm=nn.BatchNorm2d):
        super(SPP, self).__init__()
        self.features = []
        self.bins = bins
        for bin in self.bins:
            self.features.append(nn.Sequential(
                # nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                norm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.conv1 = conv(in_dim, in_dim, 1, norm=norm)
        self.conv2 = conv(reduction_dim * len(bins) + in_dim, reduction_dim, 1, norm=norm)

    def forward(self, x):
        x_size = x.size()[2:]
        ar = x_size[1] / x_size[0]
        x = self.conv1(x)
        out = [x]
        for i, f in enumerate(self.features):
            grid_size = (self.bins[i], max(1, round(ar * self.bins[i])))
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            out.append(F.interpolate(f(x_pooled), x_size, mode='bilinear', align_corners=True))

        out = torch.cat(out, 1)
        out = self.conv2(out)

        return out


class singleUp(nn.Module):
    def __init__(self, in_plane, out_plane=128, scale=1, norm=nn.BatchNorm2d):
        super(singleUp, self).__init__()
        self.scale = scale
        self.blend_conv = nn.Sequential(nn.Conv2d(in_plane, out_plane, 3, padding=1, bias=False),
                                        norm(out_plane))

    def forward(self, x):
        x = self.blend_conv(x)
        # x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x


class Upsample(nn.Module):
    def __init__(self, skip_planes, out_planes=128, norm=nn.BatchNorm2d):
        super(Upsample, self).__init__()
        self.use_skip = True
        self.bottleneck = conv(skip_planes, out_planes, 1, 0, norm=norm)
        self.blend_conv = conv(out_planes, out_planes, 3, 1, norm)

    def forward(self, skip, x):
        skip = self.bottleneck(skip)
        skip_size = skip.size()[2:4]
        x = F.interpolate(x, skip_size, mode='bilinear', align_corners=False)
        if self.use_skip:
            x = x + skip
        x = self.blend_conv(x)
        return x


if __name__ == '__main__':
    model = ATONet(classes=19).cuda()

    print(model)
