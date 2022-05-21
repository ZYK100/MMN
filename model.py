import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        pool_dim = 2048
        self.l2norm = Normalize(2)        
        self.bottleneck1 = nn.BatchNorm1d(pool_dim)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)
        self.classifier1 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier1.apply(weights_init_classifier)
        
        self.bottleneck2 = nn.BatchNorm1d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)
        self.classifier2 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier2.apply(weights_init_classifier)
        
        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3.apply(weights_init_kaiming)
        self.classifier3 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier3.apply(weights_init_classifier)
        
        self.bottleneck4 = nn.BatchNorm1d(pool_dim)
        self.bottleneck4.bias.requires_grad_(False)  # no shift
        self.bottleneck4.apply(weights_init_kaiming)
        self.classifier4 = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier4.apply(weights_init_classifier)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encode1 = nn.Conv2d(3, 1, 1)
        self.encode1.apply(my_weights_init)
        self.fc1 = nn.Conv2d(1, 1, 1)
        self.fc1.apply(my_weights_init)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn1.apply(weights_init_kaiming)

        
        self.encode2 = nn.Conv2d(3, 1, 1)
        self.encode2.apply(my_weights_init)
        self.fc2 = nn.Conv2d(1, 1, 1)
        self.fc2.apply(my_weights_init)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn2.apply(weights_init_kaiming)


        self.decode = nn.Conv2d(1, 3, 1)
        self.decode.apply(my_weights_init)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            gray1 = F.relu(self.encode1(x1))
            gray1 = self.bn1(F.relu(self.fc1(gray1)))

            gray2 = F.relu(self.encode2(x2))
            gray2 = self.bn2(F.relu(self.fc2(gray2)))            
            
            gray = F.relu(self.decode(torch.cat((gray1, gray2),0)))
            
            gray1, gray2 = torch.chunk(gray, 2, 0)
            xo = torch.cat((x1, x2), 0)

            x1 = self.visible_module(torch.cat((x1, gray1),0))
            x2 = self.thermal_module(torch.cat((x2, gray2),0))

            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            gray1 = F.relu(self.encode1(x1))
            gray1 = self.bn1(F.relu(self.fc1(gray1)))
            gray1 = F.relu(self.decode(gray1))

            x = self.visible_module(torch.cat((x1, gray1),0))
        elif modal == 2:
            gray2 = F.relu(self.encode2(x2))
            gray2 = self.bn2(F.relu(self.fc2(gray2)))
            gray2 = F.relu(self.decode(gray2))

            x = self.thermal_module(torch.cat((x2, gray2),0))


        # shared block
        x = self.base_resnet.base.layer1(x)
        x = self.base_resnet.base.layer2(x)
        x = self.base_resnet.base.layer3(x)
        x = self.base_resnet.base.layer4(x)
        x41, x42, x43, x44 = torch.chunk(x, 4, 2)
        
        x41 = self.avgpool(x41)
        x42 = self.avgpool(x42)
        x43 = self.avgpool(x43)
        x44 = self.avgpool(x44)
        x41 = x41.view(x41.size(0), x41.size(1))
        x42 = x42.view(x42.size(0), x42.size(1))
        x43 = x43.view(x43.size(0), x43.size(1))
        x44 = x44.view(x44.size(0), x44.size(1))

        feat41 = self.bottleneck1(x41)
        feat42 = self.bottleneck2(x42)
        feat43 = self.bottleneck3(x43)
        feat44 = self.bottleneck4(x44)

        if self.training:
            return x41, x42, x43, x44, self.classifier1(feat41), self.classifier2(feat42), self.classifier3(feat43), self.classifier4(feat44), [xo, gray]
        else:
            return self.l2norm(torch.cat((x41, x42, x43, x44),1)), self.l2norm(torch.cat((feat41, feat42, feat43, feat44),1))
            
            
            
            
            
            
            
            
            
            