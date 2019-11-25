import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import cv2
import h5py
import numpy as np
from   torch.autograd import Variable
import torch
from   torch import nn
import torch.nn.functional as f
import os
from   random import randint
from   scipy.signal import find_peaks_cwt
from   cyclicLoss   import cyclicLoss

def getDataPaths(DataROOT):
    trainPath  = os.path.join(DataROOT, 'train.h5') 
    testPath   = os.path.join(DataROOT, 'test.h5' )  
    validPath  = os.path.join(DataROOT, 'valid.h5' )    
    return trainPath,testPath,validPath    

def openHDF5(rootPath):
    print("\n ==> Open HDF5 files ... ")
    f         = h5py.File(rootPath, 'r')
    data      = np.asarray(f['data'])
    label     = np.asarray(f['label'])
    return      data,label 

def cleanVect(data):
    listPat = []
    for p in xrange(data.shape[0]):  
        listSeq = []
        for i in xrange(data.shape[1]):
            if data[p][i] > 0.5:
                listSeq.append(data[p][i])
        listPat.append(np.asarray(listSeq))
    return np.asarray(listPat)  

def getPlotPlotIMTDiam (list,labelTag,pointsX,pointsY,meanT,nTimeStep,savePathFig):
    fig         = plt.figure()
    ax          = fig.add_subplot(111)
    ax.set_title ("Time series")
    for j in xrange(nTimeStep):
        ax.axvline(j*meanT, color='k', linestyle='--')
    ax.plot      (list, '-', label=labelTag,color='g')
    ax.scatter(pointsX, pointsY, c='blue', s=200.0, label='peak',
            alpha=0.3, edgecolors='none')
    ax.set_xlabel('Cardiac time [s]')
    ax.set_ylabel("Diam [mm]")
    ax.legend()
    fig.savefig(savePathFig)

def getRandomInt(getIndex):
    return randint(0, getIndex)

def getMeanT(Label):
    x          = find_peaks_cwt(Label, np.arange(1, 10))
    peaks      = Label[x]
    TList      = []
    for i in xrange(len(x)-1):
        Tdelta = np.linalg.norm(x[i+1]-x[i])
        TList.append(Tdelta) 
    Taverage   = np.mean(TList) 
    nTimeStep  = int(len(Label)/Taverage)
    return x,peaks,\
           Taverage,nTimeStep

def getData(root):
    trainPath,\
    testPath,\
    validPath              = getDataPaths (root)
    dataTrain,labelTrain   = openHDF5     (trainPath           )
    dataTest,labelTest     = openHDF5     (testPath            )
    dataValid,labelValid   = openHDF5     (validPath           )
    numIndex               = getRandomInt (labelTrain.shape[0]-1)
    data                   = cleanVect(labelTrain[numIndex])
    getRandaIMT            = data   [0]
    getRandDiam            = data   [1]
    return getRandDiam,\
           getRandaIMT

if __name__ == "__main__":

    print("\n ==> Test Periodic Loss ... ")
    
    ROOT             = "/data/ns14/dataset/ultrasound/split" 
    SaveImage        = "/home/ns14/ultrasound-project/code/dataset/PeriodicLoss/periodicLoss"
    
    print("\n ==> Extract random value from train-set ... \n")
    
    LoadDiam,LoadIMT = getData(ROOT)

    print("\n ==> Genrate Random input as target lenght ...")

    input_from_CNN  = np.random.rand(1, len(LoadDiam)) [0]


    input  = np.asarray(input_from_CNN, dtype=np.float32)
    target = np.asarray(LoadDiam, dtype=np.float32)

    print("\n ==> Create variable from input and target")
    
    # input require gradient becasuse comes from 
    input  = Variable(torch.from_numpy(input),requires_grad=True)
    target = Variable(torch.from_numpy(target))
    
    print(input.size() [0])
    print(target.size()[0])

    print("\n ==> Estimation of the average value of the target")

    x,y,Taverage,\
    nTimeStep        = getMeanT(LoadDiam)
    

    print("\n ==> T mean is: "        + str(Taverage))
    print("\n ==> N time steps are: " + str(nTimeStep))
    print("\n ==> total: " + str(int(Taverage*nTimeStep)))
    print("\n ==> lenght signal is : "+ str(len(LoadDiam)))


    print("\n ==> Save image of the signal: " + str(SaveImage)+".jpg" )
    
    getPlotPlotIMTDiam (LoadDiam,"Diam",x,y,Taverage,nTimeStep,SaveImage)

    print("Create cyclic Loss: ")

    loss  = cyclicLoss(0.5)

    error = loss(input,target,Taverage,nTimeStep)

    error.backward()

    print("\n ==> Error: "+ str(error))




     














     
  








