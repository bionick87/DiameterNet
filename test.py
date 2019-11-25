###############################################
## Nicolo Savioli, PhD King's Collage London ##
###############################################

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from   model import thicknessnet
from loder import loder
import os 
import numpy as np 
from train import Train

if __name__ == "__main__":
    #alexnet,densenet,inception,resnet,vgg
    typeModel      = 'inception'
    #copyframe,unidir,bidir
    typeGRU        = 'copyframe'
    frameSecond    =  5
    num_class      =  2
    in_channels    =  1
    batchNorm      =  True
    sizeImage      =  128
    lr             =  1e-4
    rootPath       =  "/data/ns14/dataset/ultrasound/split"
    SavePath       =  "/data/ns14/ultrasound-results"
    NumEpochs      =  2
    trian          =  Train(num_class,in_channels,typeModel,\
                           typeGRU,sizeImage,batchNorm,lr,\
                           frameSecond,NumEpochs,rootPath,SavePath) 
    trian.getTrain()


