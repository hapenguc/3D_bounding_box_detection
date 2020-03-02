import os
import torch
import torch.nn as nn

class VGG(nn.Module):

def __init__(self,x):

    self.bin = 2
    self.flatten = (512/32)*2
    self.layer1 = self.block_layer(3, 64, 2)
    self.layer2 = self.block_layer(64, 128, 2)
    self.layer3 = self.block_layer(128, 256, 3)
    self.layer4 = self.block_layer(256, 512, 3)
    self.layer5 = self.block_layer(512, 512, 3)

    self.dense_dim_1 = nn.Linear(self.flatten*512, 512)
    self.dense_dim_2 = nn.Linear(512, 3)

    self.dense_ori_1 = nn.Linear(self.flatten*512,256)
    self.dense_ori_2 = nn.Linear(256, self.bin*2)
    #elf.lambda = nn.Sequential(Lambda(lambda x: 
    
    self.dense_con_1 = nn.Linear(self.flatten*512, 256)
    self.dense_con_2 = nn.Linear(256, self.bin)

    self.leakyrelu = nn.LeakyReLU(0.1)
    self.dropout = nn.Dropout(0.5)
    

    
def block_layer(self, input_channels, output_channels, basic_num):
    layers = []
    for i in range(basic_num):
        if i == 0:
           layers.append(nn.Sequential(
                nn.Conv2d(in_channels = input_channels, out_channels = ouput_channels, kernel_size = 3, stride = 1,bias = False),
                nn.ReLU(inplace = True)))

        else:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels = ouput_channels, out_channels = ouput_channels, kernel_size = 3, stride = 1,bias = False),
                nn.ReLU(inplace = True)))
    layers.append(nn.MaxPool2d(kernel_size = (2,2) stride = (2,2))
    
    return nn.Sequential(*layers)


def forward(self,x):

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)

    x = x.view(-1)
    
    dimensions = self.dense_dim_1(x)
    dimensions = self.leakyrelu(dimensions)
    dimensions = self.dropout(dimensions)
    dimensions = self.dense_dim_2(dimensions)
    dimensions = self.leakyrelu(dimensions)

    orientation = self.dense_ori_1(x)
    orientation = self.leakyrelu(orientation)
    orientation = self.dropout(orientation)
    orientation = self.dense_ori_2(orientation)
    orientation = self.leakyrelu(orientation)
    orientation = orientation.reshape(self.bin,-1)
    !!!!ntation = self.lambda(orientation)

    confidence = self.dense_con_1(x)
    confidence = self.leakyrelu(confidence)
    confidence = self.dropout(confidence)
    confidence = self.dense_con_2(confidence)


def network():
  model = VGG()
  return model