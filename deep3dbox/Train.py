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
from torchvision.models import vgg
from torch.autograd import Variable
import setproctitle
setproctitle.setproctitle("centernet-3d-lyc")


if __name__ == "__main__":
    wolk_path = os.path.abspath(os.path.dirname(__file__))
    model_path = wolk_path + "/pkl"
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    
    model_list = [x for x in sorted(os.listdir(model_path)) if x.endswith(".pkl")]
     
    with open(wolk_path + "/config.yaml","r") as f:
        config = yaml.load(f)
    path = config["kitti_path"]
    epochs = config["epochs"]
    batches = config["batches"]
    bins = config["bins"]
    alpha = config["alpha"]
    w = config["w"]
      
    print("load train data!")
    print("load val data!")  

    data = Dataset.ImageDataset(path + "/training")
    #print("data:")
    data = Dataset.BatchDataset(data, batches, bins)

    
    if len(model_list) == 0:
        print("No previous model found, start training!")
        vgg = vgg.vgg19_bn(pretrained = True)
        model = Model.Model(features = vgg.features, bins = bins).cuda()
    else:
        print("Find previous model %s" %model_list[-1])
        vgg = vgg.vgg19_bn(pretrained = False)
        model = Model.Model(features = vgg.features, bins = bins).cuda()
        param = torch.load(model_path + "/%s"%model_list[-1])
        model.load_state_dict(param)

    
    opt_SGD = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    dim_LossFunc = nn.MSELoss().cuda()
    conf_LossFunc = nn.CrossEntropyLoss().cuda()
    #print("33333",float(data.num_of_patch)/batches)
    iter_each_time = round(float(data.num_of_patch)/batches)
    for epoch in range(epochs):
        for i in range(int(iter_each_time)):
            batch,confidence,confidence_multi, angleDiff, dimGT = data.Next()
            #print("batch:",batch.shape)  #[4,3,224,224]
            #print("confidence:",confidence)
            confidence_arg = np.argmax(confidence, axis = 1)
            #print("confidence_arg:",confidence_arg)
            batch = Variable(torch.FloatTensor(batch), requires_grad = False).cuda()
            confidence = Variable(torch.LongTensor(confidence.astype(np.int)), requires_grad = False).cuda()
            confidence_multi = Variable(torch.LongTensor(confidence_multi.astype(np.int)),requires_grad = False).cuda()
            angleDiff = Variable(torch.FloatTensor(angleDiff), requires_grad = False).cuda()
            dimGT = Variable(torch.FloatTensor(dimGT), requires_grad = False).cuda()
            confidence_arg = Variable(torch.LongTensor(confidence_arg.astype(np.int)),requires_grad = False).cuda()
       
            [orient, conf, dim] = model(batch)
            #print("conf:",conf)
            #print("confidence_arg:",confidence_arg)
            conf_loss = conf_LossFunc(conf,confidence_arg)
            orient_loss = Model.OrientationLoss(orient, angleDiff, confidence_multi)
            dim_loss = dim_LossFunc(dim,dimGT)
            loss_theta = conf_loss + w*orient_loss
            #loss = alpha + dim_loss + loss_theta
            loss = dim_loss + loss_theta

       
            if i % 10 == 0:
                c_l = conf_loss.cpu().data.numpy()
                o_l = orient_loss.cpu().data.numpy()
                d_l = dim_loss.cpu().data.numpy()
                t_l = loss.cpu().data.numpy()
                now = datetime.datetime.now()
                now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
                print("-------%s Epoch %lf --------"%(now_s,epoch))
                print("Confidence Loss %lf ------"%c_l)
                print("Orientation Loss %lf ------"%o_l)
                print("Dimension Loss: %lf"%d_l)
                print("Total Loss: %lf"%t_l)
            if i%500 == 0:
                now = datetime.datetime.now()
                now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
                name = model_path +"/model_%s.pkl"%now_s
                #torch.save(model.state_dict(), name)
          
            
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
    now = datetime.datetime.now()
    now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
    name = model_path +"/model_%s.pkl"%now_s
    torch.save(model.state_dict(), name)

    print("model_savepath:",model_path)
