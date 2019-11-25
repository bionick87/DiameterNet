import torch.nn.functional as F
import torch
from   torch.autograd import Variable
import torch.nn as nn
import numpy as np


class SmoothPeakLoss(torch.nn.Module):
  
    def __init__(self,weight):
        super(SmoothPeakLoss,self).__init__()
        self.weight    = weight
        self.sumAccum  = 0
     
    def checkValue(self,value,peaks):
        valCheck = False
        for i in xrange(peaks.size()[0]):
            if value.data[0] == peaks[i].data[0]:
                valCheck = True
            else:
                valCheck = False
        return valCheck

    def forward(self,input,\
                target,peaks):
        accTensor = torch.ones(input.size()[0])
        sumAccum  = 0
        for i in xrange(target.size()[0]): 
            if self.checkValue(target[i],peaks) == True:
               sumAccum += (input[i]-target[i])**2
            else:    
               sumAccum += self.weight*(input[i]-target[i])**2 
        return sumAccum/input.size()[0]
  
        



