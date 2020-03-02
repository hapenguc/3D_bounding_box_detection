import os
import sys
import cv2
import torch
from torchvision.models import vgg
import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#import ipdb


#def OrientationLoss(orient, orientGT)

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim = 1)[1]
    
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1*torch.cos(theta_diff - estimated_theta_diff).mean()
"""
def OrientationLoss(orient, angleDiff, confGT):
    #
    # orid = [sin(delta), cos(delta)] shape = [batch, bins, 2]
    # angleDiff = GT - center, shape = [batch, bins]
    #
    #print("orient;",orient)
    [batch, _, bins] = orient.size()
    cos_diff = torch.cos(angleDiff)
    sin_diff = torch.sin(angleDiff)
    cos_ori = orient[:, :, 0]
    sin_ori = orient[:, :, 1]
    mask1 = (confGT != 0)
    mask2 = (confGT == 0)
    count = torch.sum(mask1, dim=1)
    tmp = cos_diff * cos_ori + sin_diff * sin_ori
    #print("tmp;",tmp)
    tmp[mask2] = 0
    total = torch.sum(tmp, dim = 1)
    count = count.type(torch.FloatTensor).cuda()
    total = total / count
    #print("total:",total)
    return torch.sum(total) / batch
"""
class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) # to get sin and cos
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins)
                    #nn.Softmax()
                    #nn.Sigmoid()
                )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)
        #print("orientation:",orientation.shape) #4
        orientation = orientation.view(-1, self.bins, 2)
        #print("orientation:",orientation.shape) #[2,2]
        orientation = F.normalize(orientation, dim=2)
        #print("orientation:",orientation.shape)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension
