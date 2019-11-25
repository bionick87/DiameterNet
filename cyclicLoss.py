########################################################
## Nicolo Savioli, PhD student King's Collage London  ##
########################################################

import torch.nn.functional as F
import torch
from   torch.autograd import Variable
import torch.nn as nn
import numpy as np

class cyclicLoss(torch.nn.Module):
    def __init__(self,lamda):
        super(cyclicLoss,self).__init__()
        self.lamda     = lamda
        self.MSE       = torch.nn.MSELoss(size_average=True) 

    def forward(self,input,target,Taverage,nSteps):
        T        = int(Taverage)
        N        = int(nSteps)
        sumAccum = 0
        nomL1    = 0
        if nSteps != 1:
            for n in xrange(1,N): 
                for j in xrange(0,T):
                    sumAccum += torch.pow(input[j+(n-1)*T]-input[j+n*T], 2)
            nomL1 = self.lamda*torch.sqrt(sumAccum)
        totCyclicLoss =  self.MSE(input,target) + nomL1
        return totCyclicLoss


            
