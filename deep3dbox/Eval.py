import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2
import yaml
import time
import datetime

import Model
import Dataset
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg



def save_result(info, Alpha, rot_y):
    ImageID = info['ID']
    Class = info['Class']
    dimGT = info['Dimension']
    w = str(dimGT[0])
    h = str(dimGT[1])
    l = str(dimGT[2])
    BOX_2D = info['Box_2D']
    left_x = str(BOX_2D[0][0])
    left_y = str(BOX_2D[0][1])
    right_x = str(BOX_2D[1][0])
    right_y = str(BOX_2D[1][1])
    Loc = info['Location']
    location_x = str(Loc[0])
    location_y = str(Loc[1])
    location_z = str(Loc[2])
    Alpha = round(Alpha, 2)
    rot_y = round(rot_y, 2)

    
 
    save_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc_mask/deep3dbox/output/"
    save_path = save_path + ImageID + ".txt"
    with open(save_path, "a", encoding = 'utf-8') as f:

        line = Class + " 0.00 " + "0 " + str(Alpha) + " " + left_x + " " + \
            left_y + " " + right_x + " " + right_y + " " + w + \
            " " + h + " " + l + " " + location_x + " " \
            + location_y + " " + location_z + " " + str(rot_y) + " 0.80 " + "\n"

        f.writelines(line)
        f.close()

if __name__ == '__main__':
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/model'
    pkl_path = os.path.abspath(os.path.dirname(__file__)) + '/pkl'
    if not os.path.isdir(store_path):
        print ('No folder named \"models/\"')
        exit()

    model_lst = [x for x in sorted(os.listdir(pkl_path)) if x.endswith('.pkl')]

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    path = config['kitti_path']
    epochs = config['epochs']
    batches = config['batches']
    bins = config['bins']
    alpha = config['alpha']
    w = config['w']

    data = Dataset.ImageDataset(path + '/training')
    data = Dataset.BatchDataset(data, batches, bins, mode = 'eval')
    
    if len(model_lst) == 0:
        print ('No previous model found, please check it')
        exit()
    else:
        print ('Find previous model %s'%model_lst[-1])
        vgg = vgg.vgg19_bn(pretrained=False)
        model = Model.Model(features=vgg.features, bins=bins).cuda()
        params = torch.load(pkl_path + '/%s'%model_lst[-1])
        model.load_state_dict(params)
        model.eval()

    angle_error = []
    dimension_error = []
    for i in range(data.num_of_patch):
        batch, centerAngle, info = data.EvalBatch()
        #print("info:",info)
        dimGT = info['Dimension']
        Ry = info['Ry']
        batch = Variable(torch.FloatTensor(batch), requires_grad=False).cuda()


        [orient, conf, dim] = model(batch)
        orient = orient.cpu().data.numpy()[0, :, :]
        #print("orient:",orient)
        conf = conf.cpu().data.numpy()[0, :]
        #print("conf:",conf)
        dim = dim.cpu().data.numpy()[0, :]
        #print("dim:",dim)
        argmax = np.argmax(conf)
        #print("argmax:",argmax)
        orient = orient[argmax, :]
        #print("orient_new:",orient)
        cos = orient[0]
        #print("cos:",cos)
        sin = orient[1]
        #print("sin:",sin)

        theta = np.arctan2(sin, cos)
        #print("theta:",theta)
        if argmax == 0:
            alpha_test = theta + np.pi/2
        else:
            alpha_test = theta + 3*np.pi/2
        alpha_test = alpha_test - np.pi
        #print("******************************alpha_test:",alpha_test)
        #print("info:",info)
        ImageID = info['ID']
        BOX_2D = info['Box_2D']
        left_x = BOX_2D[0][0]
        right_x = BOX_2D[1][0]
        #print("left:",left_x)
        #print("right:",right_x)
        center_x = left_x + ((right_x - left_x)/2)
        #print("center_x:",center_x)
        cx = 6.040814000000e+02
        fx = 7.070493000000e+02 
        

        Dim = info['Dimension']
        Loc = info['Location']
        Alpha = info['Alpha']
        Loc_x,Loc_y,Loc_z = Loc[0],Loc[1],Loc[2]
        
        rot_y = alpha_test + np.arctan2(center_x - cx, fx)
        if rot_y > np.pi:
            rot_y -= 2 * np.pi
        if rot_y < -np.pi:
            rot_y += 2 * np.pi
        
        #print("****************R_y_det:",rot_y)
        #print("****************R_y:",Ry/180*np.pi)
        save_result(info, alpha_test, rot_y)


            


        



