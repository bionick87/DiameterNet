########################################################
## Nicolo Savioli, PhD student King's Collage London  ##
########################################################

import cv2
import h5py
import numpy as np
from   torch.autograd import Variable
import torch
from   torch import nn
import torch.nn.functional as f
from   scipy.signal import find_peaks_cwt
from   peakdetect import peakdetect

class loder():

    def __init__(self,pathData,\
                 nFrameSecond,typeLoss,typeDataset):
        self.rootPath                      = pathData
        self.typeDataset                   = typeDataset
        if self.typeDataset == "Synthetic":
           self.data,self.labels,\
           self.period                     = self.openHDF5_Synthetic()
        else:
           self.data,self.labels           = self.openHDF5()
        self.getMeanStdTrain()
        self.data                          = self.dataNormalisation(self.data)
        self.labels                        = self.getLabelTrap()
        self.nPat,self.nSeqs,self.sizeImgs = self.getdim()  
        self.nFrameSecond                  = nFrameSecond 
        self.typeLoss                      = typeLoss
        # utils:
        if self.nFrameSecond != 0:
            self.checkSplitFrame()
    
    def cleanZerOutlier(self,input):
        inputSeq = []
        for seq in xrange(len(input)):
            if input[seq] > 0.0:
               inputSeq.append(input[seq])
        return inputSeq

    def getBatchDim(self):
        return self.data.shape[0]*self.data.shape[1]
         
    def checkSplitFrame(self):
        if self.nSeqs%self.nFrameSecond != 0:
           print("ERROR: number frames should be divisible for: " + str(self.nSeqs))
    
    def getMeanStdTrain(self):
        print("==> Location path is: " +  self.rootPath)
        print("==> The Mean value is: " + str(np.mean(self.data)))
        print("==> The Std value  is: " + str(np.std(self.data)))

    def dataNormalisation(self,data):
         norm = np.divide(np.subtract(data,15.1637),25.392)
         return norm 

    def getLabelTrap(self):
        tpLabels = self.labels.transpose((0,2,1))
        return tpLabels

    def getdim(self):
        dim      = self.data.shape
        nPat     = dim[0]
        nSeqs    = dim[1]
        sizeImgs = dim[3]
        return nPat,nSeqs,sizeImgs

    def vflip(self,img):
        vimg = cv2.flip(img,1)
        return vimg

    def hflip(self,img):
        himg = cv2.flip(img,0)
        return himg
    
    def getRandomIndex(self,sample):
        randImg = int(np.random.uniform(0,sample,1))
        return  randImg

    def reshapeTestSeqs(self,SeqData,SeqLabel):
        numWinds    = int(SeqData.shape[1]/self.nFrameSecond)        
        NewSeqData  = np.reshape(SeqData,(SeqData.shape [0],numWinds, self.nFrameSecond,1,\
                                          SeqData.shape [3],SeqData.shape[4]))
        NewSeqLabel = np.reshape(SeqLabel,(SeqData.shape[0],numWinds,self.nFrameSecond,\
                                          SeqLabel.shape[2]))
        return NewSeqData,NewSeqLabel
    
    def reshapeTrainSeqs(self,SeqData,SeqLabel):
        numWinds = int(SeqData.shape[0]/self.nFrameSecond)        
        NewSeqData  = np.reshape(SeqData,(numWinds, self.nFrameSecond,1,\
                                 SeqData.shape[2],SeqData.shape[3]))
        NewSeqLabel = np.reshape(SeqLabel,(numWinds,self.nFrameSecond,\
                                 SeqLabel.shape[1]))
        return NewSeqData,NewSeqLabel

    def getDataArgum(self,img,index):
        outputImage = None
        if   index == 0:
            vimg        = cv2.flip(img,1)
            outputImage = cv2.flip(vimg,0)
        elif index == 1:
            himg        = cv2.flip(img,0)
            outputImage = cv2.flip(himg,1)          
        elif index == 2:
            outputImage = cv2.flip(img,1)
        elif index == 3:
            outputImage = cv2.flip(img,0)
        elif index == 4:
            outputImage = img
        return outputImage 
    
    def openHDF5(self):
        print("\n ==> Open HDF5 files ... ")
        f         = h5py.File(self.rootPath, 'r')
        data      = np.asarray(f['data'])
        label     = np.asarray(f['label'])
        return  data,label 

    def openHDF5_Synthetic(self):
        print("\n ==> Open synthetic HDF5 files ... ")
        f         = h5py.File(self.rootPath, 'r')
        data      = np.asarray(f['data'])
        label     = np.asarray(f['label'])
        period    = np.asarray(f['period']).transpose((0,2,1))
        return  data,label,period

    def getTrainSeqs(self):
        SeqData,SeqLabel       = self.getSeqs(self.data,self.labels)
        return SeqData,SeqLabel
    
    def getTestSeqs(self):
        NewSeqData,NewSeqLabel = self.reshapeTestSeqs(self.data,self.labels)
        return NewSeqData,NewSeqLabel 

    def getRadomFrames(self,SeqData,SeqLabels):
        randFrame              = self.getRandomIndex(SeqData.shape[0])
        return SeqData[randFrame],SeqLabels[randFrame]

    def getRadomFramesForSin(self,SeqData,SeqLabels,SeqPeriod):
        randFrame              = self.getRandomIndex(SeqData.shape[0])
        return SeqData[randFrame],SeqLabels[randFrame],SeqPeriod[randFrame]

    def getArgFrames(self,SeqData,getRandArgIndex):
        loadData               = np.zeros((SeqData.shape[0],1,self.sizeImgs,self.sizeImgs))
        for i in xrange(SeqData.shape[0]):
            loadData [i][0]    = self.getDataArgum(SeqData[i][0],getRandArgIndex)
        return loadData

    def loadingTrain(self):
        SeqOutData       = None
        SeqOutLabels     = None
        if self.typeDataset == "Synthetic":
           SeqDataOut    = self.data
           SeqLabelOut   = self.labels
           SeqPeriodOut  = self.period 
        else:
           SeqDataOut    = self.data
           SeqLabelOut   = self.labels
           SeqPeriodOut  = 0 
        if self.nFrameSecond != 0:
            randPat               = self.getRandomIndex(self.nPat)
            SeqData               = SeqDataOut [randPat]
            SeqLabel              = SeqLabelOut[randPat]
            #Reshape
            SeqData,SeqLabel      = self.reshapeTrainSeqs(SeqData,SeqLabel)
            Seq_Data,SeqOutLabels = self.getRadomFrames(SeqData,SeqLabel)
            getRandArgIndex       = self.getRandomIndex(5)
            SeqOutData            = self.getArgFrames(Seq_Data,getRandArgIndex)
        else:
            if self.typeDataset == "Synthetic":
                SeqOutData,\
                SeqOutLabels,\
                SeqPeriodOut           = self.getRadomFramesForSin(SeqDataOut,\
                                                     SeqLabelOut,SeqPeriodOut)
                getRandArgIndex        = self.getRandomIndex(5)
                SeqOutData             = self.getArgFrames(SeqOutData,getRandArgIndex)      
            else:
                SeqOutData,\
                SeqOutLabels           = self.getRadomFrames(SeqDataOut,SeqLabelOut)
                getRandArgIndex        = self.getRandomIndex(5)
                SeqOutData             = self.getArgFrames(SeqOutData,getRandArgIndex)

        return SeqOutData,\
               SeqOutLabels,\
               SeqPeriodOut  

    def loadingTest(self):
        SeqData,SeqLabel    = self.getTestSeqs()
        return SeqData,SeqLabel  
    
    def getCUDADataVariable(self,data):
        dataVar   = Variable(torch.from_numpy(np.asarray(data,dtype=np.float32))) 
        dataVar   = dataVar.cuda ()
        return dataVar

    def getCUDATargetVariable(self,label):
        labelVar  = Variable(torch.from_numpy(np.asarray(label,dtype=np.float32)))
        labelVar  = labelVar.cuda()
        return labelVar

    def getCUDAPeaks(self,peaks):
        peaks  = peaks.cuda()
        return peaks

    def getPeaksVariable(self,peakList):
        npArray = np.asarray(peakList)
        thArray = torch.from_numpy(npArray)
        return Variable(thArray)
     
    def formatPeaksList(self,listPoints):
        listX = []
        listY = []
        for points in listPoints:
            for point in points:
                listX.append(point[0])
                listY.append(point[1])
        return self.getPeaksVariable(listX),\
               self.getPeaksVariable(listY)

    def getMeanT(self,Label):
        x          = find_peaks_cwt(Label,np.arange(1,2))
        TList      = []
        for i in xrange(len(x)-1):
            Tdelta = np.linalg.norm(x[i+1]-x[i])
            TList.append(Tdelta) 
        Taverage   = np.mean(TList) 
        nTimeStep  = int(len(Label)/Taverage)
        return Taverage,nTimeStep

    def getPeaks(self,loadLabels):
        peaksValuesVariable = None
        Taverage            = 0
        nTimeStep           = 0
        if self.typeLoss   == "MSEPeak":
            indexPeaksVariable,\
            peaksValuesVariable = self.formatPeaksList(peakdetect(loadLabels,lookahead=1))
        elif self.typeLoss == "MSECyclic":
            Taverage,nTimeStep  = self.getMeanT(loadLabels)
        return peaksValuesVariable,\
               Taverage,nTimeStep
    
    def getDiamAndImT(self,Labels):
        IamtList = []
        DiamList = []
        for label in Labels:
            IamtList.append(label[0])
            DiamList.append(label[1])
        return IamtList,DiamList

    def getTest(self): 
        loadData   = None
        loadLabels = None
        if self.nFrameSecond != 0:
            loadData,\
            loadLabels = self.loadingTest()                           
        else:
            loadData   = self.data
            loadLabels = self.labels
        return loadData,loadLabels                                         
                             
    def getDtataWithoutZero(self,Imgs,LablelDiam,LabelIamt):
        numZeroDiam         = self.contNotZero(LablelDiam)
        numZeroImat         = self.contNotZero(LabelIamt)
        getMinNotZeroValues = min(numZeroDiam,numZeroImat)
        NewImgList          = []
        NewLablelDiamList   = []
        NewLabelIamtList    = []
        for i in xrange(getMinNotZeroValues):
            NewImgList.append        (Imgs[i]      )
            NewLablelDiamList.append (LablelDiam[i])
            NewLabelIamtList.append  (LabelIamt [i])
        return NewImgList,\
               NewLablelDiamList,\
               NewLabelIamtList

    def contNotZero(self,input):  
        cont = 0 
        for seq in xrange(len(input)):
            if input[seq] > 0.0:
               cont += 1
        return cont                    
        
    def cleanZerOutlier(self,input):
        inputSeq = []
        for seq in xrange(len(input)):
            if input[seq] > 0.0:
               inputSeq.append(input[seq])
        return inputSeq

    def getTrain(self):      
        ####################################################
        loadData      = loadTrgetDiam = loadTrgetIamt = None
        loadPeaksDiam = loadPeaksIamt = TaverageDiam  = None
        TaverageIamt  = nTimeStepDiam = nTimeStepIamt = None
        ####################################################                                              
        loadData,loadLabels,\
        SeqPeriodOut                  = self.loadingTrain() 
        IamtList,DiamList             = self.getDiamAndImT(loadLabels)
        IamtList                      = IamtList
        DiamList                      = DiamList
        if self.typeDataset == "Synthetic":
            TaverageIamt  = TaverageDiam = SeqPeriodOut[0][0]
            nTimeStepDiam = int(loadData.shape[0]/TaverageIamt)
            nTimeStepIamt = int(loadData.shape[0]/TaverageIamt)
        else:
            loadPeaksDiam,\
            TaverageDiam,nTimeStepDiam    = self.getPeaks(DiamList)   
            # Iamt peaks and periodic metrics
            loadPeaksIamt,\
            TaverageIamt,nTimeStepIamt    = self.getPeaks(IamtList) 

        # Load Images sequence on  GPU
        loadData                      = self.getCUDADataVariable  (loadData)
        # Load Diam sequence   on  GPU
        loadTrgetDiam                 = self.getCUDATargetVariable(DiamList)
        # Load Iamt sequence   on  GPU
        loadTrgetIamt                 = self.getCUDATargetVariable(IamtList)
        if self.typeLoss   == "MSEPeak":
            # Loda Diam peaks   on  GPU
            loadPeaksDiam             = self.getCUDAPeaks(loadPeaksDiam)
            # Loda Iamt peaks  on  GPU
            loadPeaksIamt             = self.getCUDAPeaks(loadPeaksIamt)
        
        return loadData     ,loadTrgetDiam,\
               loadTrgetIamt,loadPeaksDiam,\
               loadPeaksIamt,TaverageDiam ,\
               TaverageIamt ,nTimeStepDiam,\
               nTimeStepIamt



 





