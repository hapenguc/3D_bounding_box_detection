# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.decode import ctdet_decode

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = conv3x3(512, 256, stride=1)

        if head_conv > 0:
          self.hm_layer =  nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, self.heads['hm'], 
                    kernel_size=1, stride=1, padding=0))

          self.wh_layer =  nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, self.heads['wh'], 
                    kernel_size=1, stride=1, padding=0))

          self.reg_layer =  nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, self.heads['reg'], 
                    kernel_size=1, stride=1, padding=0))

        else:
          self.hm_layer = nn.Conv2d(
                  in_channels=256,
                  out_channels=self.heads['hm'],
                  kernel_size=1,
                  stride=1,
                  padding=0)

          self.wh_layer = nn.Conv2d(
                  in_channels=256,
                  out_channels=self.heads['wh'],
                  kernel_size=1,
                  stride=1,
                  padding=0)

          self.reg_layer = nn.Conv2d(
                  in_channels=256,
                  out_channels=self.heads['reg'],
                  kernel_size=1,
                  stride=1,
                  padding=0)
        
        self.__setattr__('hm',self.hm_layer)
        self.__setattr__('wh',self.wh_layer)
        self.__setattr__('reg',self.reg_layer)
 
        num_2d = heads['hm'] + heads['wh'] + heads['reg']

        self.dddector = nn.Sequential(
                  nn.Conv2d(num_2d, 64,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(64, 256, 
                    kernel_size=1, stride=1, padding=0))

        if head_conv > 0:
          self.dep_layer =  nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, self.heads['dep'], 
                    kernel_size=1, stride=1, padding=0))

          self.rot_layer =  nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, self.heads['rot'], 
                    kernel_size=1, stride=1, padding=0))

          self.dim_layer =  nn.Sequential(
                  nn.Conv2d(256, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, self.heads['dim'], 
                    kernel_size=1, stride=1, padding=0))
        else:
          self.dep_layer = nn.Conv2d(
                  in_channels=256,
                  out_channels=self.heads['dep'],
                  kernel_size=1,
                  stride=1,
                  padding=0)

          self.rot_layer = nn.Conv2d(
                  in_channels=256,
                  out_channels=self.heads['rot'],
                  kernel_size=1,
                  stride=1,
                  padding=0)

          self.dim_layer = nn.Conv2d(
                  in_channels=256,
                  out_channels=self.heads['dim'],
                  kernel_size=1,
                  stride=1,
                  padding=0)
        self.__setattr__('dep',self.dep_layer)
        self.__setattr__('rot',self.rot_layer)
        self.__setattr__('dim',self.dim_layer)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        ret = {}
        #head{'hm','wh','reg','dep','rot','dim']

        #print("x:",x.shape) [4,256,12,40]
        hm = self.hm_layer(x)
        wh = self.wh_layer(x)
        reg = self.reg_layer(x)
        
        #print("hm",hm.shape) [4,3,12,40]
        x = torch.cat((hm,wh,reg), 1)
        #print("x:",x.shape) #[4,7,12,40]
        #print("11111",x)
        #x.shape = [7, 12,40]
        x = self.dddector(x)
        
        dep = self.dep_layer(x)
        rot = self.rot_layer(x)
        dim = self.dim_layer(x)
        ret['hm'] = hm
        ret['wh'] = wh
        ret['reg'] = reg
        ret['dep'] = dep
        ret['rot'] = rot
        ret['dim'] = dim
        #print("ret:",ret)
        return [ret]

  

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            #for _, m in self.deconv_layers.named_modules():
                #if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    #nn.init.normal_(m.weight, std=0.001)
                    #if self.deconv_with_bias:
                        #nn.init.constant_(m.bias, 0)
                #elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    #nn.init.constant_(m.weight, 1)
                    #nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                      # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                      # print('=> init {}.bias as 0'.format(name))
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net_4(num_layers, heads, head_conv):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers, pretrained=True)
  return model
