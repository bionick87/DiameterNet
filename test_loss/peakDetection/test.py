import torch.nn.functional as F
import torch
from   torch.autograd import Variable
import torch.nn as nn
from   peakdetect import peakdetect
import numpy as np 
from SmoothPeakLoss import SmoothPeakLoss



def getTensor(npList):
    npArray = np.asarray(npList)
    return Variable(torch.from_numpy(npArray))
     
def formatList(listPoints):
    listX = []
    listY = []
    for points in listPoints:
        for point in points:
            listX.append(point[0])
            listY.append(point[1])
    return getTensor(listX),getTensor(listY)


'''
def getSquare(input_np,target_np,peaks,weight):
    listAcc = []
    for i in xrange(target_np.shape[0]):
            if target_np[i] in peaks:
               listAcc.append((input_np[i] - target_np[i]))
            else:
               listAcc.append(weight*(input_np[i] - target_np[i]))
    return getTensor(listAcc)
'''


def run():
    input_np  = np.random.rand(1, 150) [0]
    target_np = np.random.rand(1, 150) [0]

    input_th  = Variable(torch.from_numpy(input_np),requires_grad=True)
    target_th = Variable(torch.from_numpy(target_np))

    # it get out in torch.DoubleTensor
    x,peaks = formatList(peakdetect(target_np, lookahead=1))

    Loss  =   SmoothPeakLoss(0.5,peaks)

    error = Loss (input_th,target_th) 


    error.backward()

    print(error)  





if __name__ == "__main__":
    run()
