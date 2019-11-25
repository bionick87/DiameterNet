###############################################
## Nicolo Savioli, PhD King's Collage London ##
###############################################

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

import cv2
import h5py
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
import torch.nn.functional as f
import os
from random import randint
#from scipy.signal import find_peaks_cwt
from peakdetect import peakdetect



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


def getPlotPlotIMTDiam (list,labelTag,pointsX,pointsY,savePathFig):
    fig         = plt.figure()
    ax          = fig.add_subplot(111)
    ax.set_title ("Time series")
    ax.plot      (list, '-', label=labelTag,color='g')
    ax.scatter(pointsX, pointsY, c='blue', s=200.0, label='peak',
            alpha=0.3, edgecolors='none')
    ax.set_xlabel('Cardiac time [s]')
    ax.set_ylabel("Diam [mm]")
    ax.legend()
    fig.savefig(savePathFig)


def getRandomInt(getIndex):
    return randint(0, getIndex)

def formatList(listPoints):
    listX = []
    listY = []
    for points in listPoints:
        for point in points:
            listX.append(point[0])
            listY.append(point[1])
    return listX,listY

     
def main(root,SaveImage):
    trainPath,\
    testPath,\
    validPath             = getDataPaths (root)
    dataTrain,labelgetRandDiamTrain  = openHDF5     (trainPath )
    dataTest,labelTest    = openHDF5     (testPath  )
    dataValid,labelValid  = openHDF5     (validPath )
    numIndex              = getRandomInt (labelTest.shape[0]-1)
    data                  = cleanVect(labelTest[numIndex])
    getRandaIMT           = data   [0]
    getRandDiam           = data   [1]

    pathSave              = os.path.join(SaveImage,"img.jpg")
    x,y                   = formatList(peakdetect(getRandDiam, lookahead=1))
    getPlotPlotIMTDiam (getRandDiam,"Diam",x,y,pathSave) 
 
if __name__ == "__main__":
    # ROOT PATH
    ROOT         = "/data/ns14/dataset/ultrasound/split" 
    SaveImage    = "/data/ns14/dataset/ultrasound/saveImgs"
    main(ROOT,SaveImage)

