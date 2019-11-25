########################################################
## Nicolo Savioli, PhD student King's Collage London  ##
########################################################

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import os 
import numpy as np 
import torch
from   torch import nn
import torch.nn.functional as f
from   torch.autograd import Variable
from   model import thicknessnet
from   loder import loder
import datetime
import shutil
import progressbar as pb
import warnings
warnings.filterwarnings("ignore")
# New Loss
from SmoothPeakLoss import SmoothPeakLoss
from cyclicLoss     import cyclicLoss

class Train():
    def __init__(self,num_class,in_channels,typeModel,\
                typeGRU,typeCriterion,dampingValue,\
                sizeImage,learning_rate,frameSecond,\
                NumEpochs,dataRealRoot,dataSyntheticRoot,\
                SavePath,typeMeasure,typeDataset):

       self.typeDataset  = typeDataset
       if   self.typeDataset == "Real":
              self.DataROOT  = dataRealRoot
       elif self.typeDataset == "Synthetic": 
              self.DataROOT  = dataSyntheticRoot
       # set variables: 
       self.num_class     = num_class 
       self.in_channels   = in_channels
       self.typeGRU       = typeGRU
       self.sizeImage     = sizeImage
       self.frameSecond   = frameSecond 
       self.learning_rate = learning_rate
       self.SavePath      = SavePath
       self.typeModel     = typeModel
       self.typeCriterion = typeCriterion
       self.dampingValue  = dampingValue
       self.typeMeasure   = typeMeasure
       #self.check()
       # build the model:
       self.criterion     = self.getCriterion                 () 
       self.MSEcritEval   = self.getEvaluationCriterion       ()
       self.ModelName     = self.getMeasurement(self.getLossName(self.getModelName()))
       self.trainPath,\
       self.testPath,\
       self.validPath     = self.getDataPaths ()
       self.model         = self.buildModel   ()
       self.optimize      = self.getOptimizer () 
       # datasets: 
       #############################################
       self.trainData     = loder(self.trainPath,\
                                  self.frameSecond,\
                                  self.typeCriterion,\
                                  self.typeDataset)
       #############################################
       self.testData      = loder(self.testPath,\
                                  self.frameSecond,\
                                  self.typeCriterion,\
                                  self.typeDataset)
       #############################################
       self.validData     = loder(self.validPath,\
                                  self.frameSecond,\
                                  self.typeCriterion,\
                                  self.typeDataset)
       #############################################
       # Iterations:
       self.NumIterations = self.trainData.getBatchDim()  
       self.NumEpochs     = NumEpochs
       self.GetInfo()
       # save folders: 
       self.modelSavePath,\
       self.modelBestPath,\
       self.trainSavePath,\
       self.testSavePath,\
       self.validSavePath = self.getFolder()
    
    def check(self):
      if self.frameSecond != 0 and self.typeCriterion != "MSE":
         print("\n ==> With 25 frames/s you can use only MSE!")
         self.typeCriterion = "MSE"
      if (self.typeCriterion == "MSEPeak" \
         or self.typeCriterion == "MSECyclic") and self.frameSecond == 25:
         print("\n ==> With MSECyclic and MSEPeak you need use more of 25 frames/s!")
         self.frameSecond   = 0

    def setProgressbar(self,description,iteration):
        widgets = [description, pb.Percentage(), ' ',
                   pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        timer   = pb.ProgressBar(widgets=widgets, maxval=iteration).start()
        return timer

    def getModelName(self):
        nameModel = None 
        if  self.typeGRU == "copyframe":
            if self.typeModel   == "alexnet":
                nameModel = "AlexNet"
            elif self.typeModel == "densenet":
                nameModel = "Densenet121"
            elif self.typeModel == "inception":
                nameModel = "InceptionV4"
            elif self.typeModel == "resnet":
                nameModel = "ResNet18"
            elif self.typeModel == "vgg":
                nameModel = "Vgg-E"
        elif self.typeGRU == "unidir":
            if self.typeModel   == "alexnet":
                nameModel = "AlexNet+GRU"
            elif self.typeModel == "densenet":
                nameModel = "Densenet121+GRU"
            elif self.typeModel == "inception":
                nameModel = "InceptionV4+GRU"
            elif self.typeModel == "resnet":
                nameModel = "ResNet18+GRU"
            elif self.typeModel == "vgg":
                nameModel = "Vgg-E+GRU"
        elif self.typeGRU == "bidir":
            if self.typeModel   == "alexnet":
                nameModel = "AlexNet+BiGRU"
            elif self.typeModel == "densenet":
                nameModel = "Densenet121+BiGRU"
            elif self.typeModel == "inception":
                nameModel = "InceptionV4+BiGRU"
            elif self.typeModel == "resnet":
                nameModel = "ResNet18+BiGRU"
            elif self.typeModel == "vgg":
                nameModel = "Vgg-E+BiGRU"
        return nameModel

    def getLossName(self,nameModel):
          nameModelCriterion = ""
          if   self.typeCriterion == "MSE":
               nameModelCriterion = nameModel + "+MSE"
          elif self.typeCriterion == "MSEPeak":
            nameModelCriterion = nameModel    + "+Smooth-Peaks-MSE"
          elif self.typeCriterion == "MSECyclic":
            nameModelCriterion = nameModel    + "+Cyclic-Peaks-MSE"
          return nameModelCriterion

    def getMeasurement(self,nameModel):
          nameModelMeasurement = ""
          if   self.typeMeasure == "Diam":
               nameModelMeasurement = nameModel + "--Diameter--" + self.typeDataset + " " + "dataset."
          if   self.typeMeasure == "Iamt":
               nameModelMeasurement = nameModel + "--Iamt--"     + self.typeDataset + " " + "dataset."
          return nameModelMeasurement

    def getTime(self):
        now = datetime.datetime.now()
        data = str(now.hour)+":"+str(now.minute)+":"+ str(now.second)+"-"+\
        str(now.day) +"-"+str(now.month)+"-"+str(now.year)
        return data
    
    def makeFolder(self,path):
        if not os.path.exists(path):
           os.makedirs(path)
    
    def getFolder(self):
        ROOTPath = os.path.join(self.SavePath,\
                                self.ModelName,self.getTime())
        modelSavePath  = os.path.join(ROOTPath,"Models") 
        modelBestPath  = os.path.join(ROOTPath,"BestModel")
        trainSavePath  = os.path.join(ROOTPath,"train")  
        testSavePath   = os.path.join(ROOTPath,"test") 
        validSavePath  = os.path.join(ROOTPath,"valid") 
        self.makeFolder  (modelSavePath)
        self.makeFolder  (modelBestPath)
        self.makeFolder  (trainSavePath)
        self.makeFolder  (testSavePath)
        self.makeFolder  (validSavePath)
        return modelSavePath,modelBestPath,\
              trainSavePath, testSavePath,\
              validSavePath
        
    def GetInfo(self):
       print("\n ==> Number of epochs is: " + str(self.NumEpochs))
       print("\n ==> Number of iteration per epoch is: " + str(self.NumIterations)) 
       print("\n ==> Model name is: " + self.ModelName)
         
    def getDataPaths(self):
        trainPath  = os.path.join(self.DataROOT, 'train.h5' ) 
        testPath   = os.path.join(self.DataROOT, 'test.h5'  )  
        validPath  = os.path.join(self.DataROOT, 'valid.h5' )    
        return trainPath,testPath,validPath    

    def buildModel(self):
        print("\n ==> Create model ...")
        model = thicknessnet(self.num_class,self.in_channels,self.typeModel,\
                self.typeGRU,self.sizeImage)
        print("\n ==> Convert model to CUDA ...")
        model = model.cuda()
        return model   

    def getCriterion(self):
        loss = None 
        if   self.typeCriterion  == "MSE":
           print ("\n ==> Standard MSE activate.")
           loss  = torch.nn.MSELoss (size_average=True)
        else self.typeCriterion == "MSECyclic":
           print ("\n ==> Cyclic MSE activate.")
           loss  = cyclicLoss       (self.dampingValue)
        print("\n ==> Convert loss to CUDA ...")
        loss = loss.cuda()
        return loss

    def getEvaluationCriterion(self):
        MSEval  = torch.nn.MSELoss (size_average=True)
        MSEval  = MSEval.cuda()
        return MSEval

    def getOptimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(),\
                                     lr=self.learning_rate)
        return optimizer
    
    # The network extract 0 class for 
    def formatValues(self,dataLoad):
        idsIamt      = Variable(torch.zeros(dataLoad.size()[0],1).long()).cuda()
        idsDim       = Variable(torch.ones(dataLoad.size()[0],1).long()).cuda()
        IamtVariable = dataLoad.gather(1, idsIamt.view(-1,1))
        DiamVariable = dataLoad.gather(1, idsDim.view(-1,1) )
        return IamtVariable,\
               DiamVariable     
    
    def getClassification(self,dataFromTrain):
        ####
        loadData                   = dataFromTrain[0]
        ####
        loadTrgetDiam              = dataFromTrain[1]
        loadTrgetIamt              = dataFromTrain[2]
        ####
        loadPeaksDiam              = dataFromTrain[3]
        loadPeaksIamt              = dataFromTrain[4]
        ####
        TaverageDiam               = dataFromTrain[5]
        TaverageIamt               = dataFromTrain[6]
        ####
        nTimeStepDiam              = dataFromTrain[7]
        nTimeStepIam               = dataFromTrain[8]
        ####
        self.optimize.zero_grad()
        # get output form CNN
        OutModel                  = self.model(loadData)
        #IamtVariable,DiamVariable  = self.formatValues(getOutModel)
        loss                        = None 
        if   self.typeCriterion     == "MSE":
              if self.typeMeasure   == "Diam":
                   Loss_diam = self.criterion(OutModel,loadTrgetDiam) 
                   loss      = Loss_diam
              elif self.typeMeasure == "Iamt":
                   Loss_iamt = self.criterion(OutModel,loadTrgetIamt)
                   loss      = Loss_iamt
        elif self.typeCriterion     == "MSEPeak":
              if self.typeMeasure   == "Diam":
                    Loss_diam = self.criterion(OutModel,loadTrgetDiam,loadPeaksDiam) 
                    loss      = Loss_diam
              elif self.typeMeasure == "Iamt":  
                    Loss_iamt = self.criterion(OutModel,loadTrgetIamt,loadPeaksIamt)
                    loss      = Loss_iamt
        elif self.typeCriterion     == "MSECyclic":
              if self.typeMeasure   == "Diam":
                    Loss_diam = self.criterion(OutModel,loadTrgetDiam,TaverageDiam,nTimeStepDiam)  
                    loss      = Loss_diam
              elif self.typeMeasure == "Iamt": 
                    Loss_iamt = self.criterion(OutModel,loadTrgetIamt,TaverageIamt,nTimeStepIam)
                    loss      = Loss_iamt
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(),5)
        self.optimize.step()
        return loss
    
    def TestCriterion(self):
        outTensor1  = Variable(torch.Tensor(25,2)).cuda()
        outTensor2  = Variable(torch.Tensor(25,2)).cuda()
        loss        = self.criterion(outTensor1,outTensor2)

    def saveRegressionLoss(self,lossList,lossSavePath):
        with open(lossSavePath, "wb") as f:
            for loss in lossList:
                f.write(str(loss) +"\n")

    def saveTestLoss(self,lossList,lossSavePath):
        with open(lossSavePath, "wb") as f:
            for data in lossList:
                f.write(str(data[0])+" "+ str(data[1]) +"\n")

    def saveValidLoss(self,lossList,lossSavePath):
        with open(lossSavePath, "wb") as f:
            for data in lossList:
                f.write(str(data) +"\n")
    
    def saveModel(self,epoch):
        print("\n ==> Save model ...")
        torch.save(self.model,\
        os.path.join(self.modelSavePath,\
        "epoch_"+str(epoch)+'.pt'))

    def getMSECurve(self,avgLoss,contEpochs,typeMode):
        fig        = plt.figure()
        ax         = fig.add_subplot(111)
        selectPath = None
        ax.set_title("Training of " + self.ModelName)
        ax.plot(contEpochs,avgLoss, '-o', label= typeMode + " MSE")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Square Error (MSE)')
        ax.legend()
        if   typeMode == "Train":
             selectPath = self.trainSavePath
        elif typeMode == "Test":
             selectPath = self.testSavePath
        elif typeMode == "Valid":
             selectPath =  self.validSavePath
        fig.savefig(os.path.join(selectPath,typeMode+"Loss.jpg"))
        self.saveRegressionLoss(avgLoss,os.path.join(selectPath,typeMode+"Loss.txt"))

    def saveBestModel(self,bestModel,typeModel):
        srcModel = os.path.join(self.modelSavePath,\
                   "epoch_"+str(bestModel)+'.pt')
        dstModel = os.path.join(self.modelBestPath,\
                   "epoch_"+str(bestModel)+"_"+typeModel+'.pt')
        shutil.copy(srcModel,dstModel)

    def deleteModelFolder(self):
        shutil.rmtree(self.modelSavePath)

    def openTxt(self,MSEfile):
        mse_list     = []
        with open(MSEfile) as f:
            mse_list = f.readlines()
        mse_list     = [float(x.strip()) for x in mse_list] 
        return mse_list

    def getMinIndexIMT(self,IMTPath): 
        openTxtIMT   =  self.openTxt(IMTPath)
        ##############################################
        TotListIMT   =  np.array([openTxtIMT])
        meanListIMT  =  np.average(TotListIMT, axis=0)
        ##############################################
        minindexIMT    =  np.argmin(meanListIMT) +1
        return minindexIMT

    def getMinIndexDiam(self,DiamPath): 
        openTxtDiam  =  self.openTxt(DiamPath)
        ##############################################
        TotListDiam  =  np.array([openTxtDiam])
        meanListDiam =  np.average(TotListDiam, axis=0)
        ##############################################
        minindexDiam   =  np.argmin(meanListDiam)+1
        return minindexDiam

    def cleanList(self,data):
        optimNums = np.count_nonzero(data)
        newList   = []
        for i in xrange(optimNums):
            newList.append(data[i])
        return newList

    def getDiamAndImT(self,Labels):
        IamtList = []
        DiamList = []
        for label in Labels:
            IamtList.append(label[0])
            DiamList.append(label[1])
        return IamtList,DiamList
    
    def cleanVect(self,data):
      listSeq    = []
      for i in xrange(len(data)):  
          if data[i] > 0.5:
              listSeq.append(data[i])
      return listSeq
    
    def getMinValid_IMT_MSE(self):
        MSEfileIMT     = os.path.join(self.validSavePath,\
                                     "ValidLossIMT.txt")
        minindexIMT = self.getMinIndexIMT(MSEfileIMT)
        return minindexIMT

    def getMinValid_Diam_MSE(self):
        MSEfileDiam    = os.path.join(self.validSavePath,\
                                     "ValidLossDiam.txt")
        minindexDiam = self.getMinIndexDiam(MSEfileDiam)
        return minindexDiam

    def getDiamEsimation(self,gtVal,\
                         predVal,typePred,p):
        typeName      =  ""
        if   typePred == "IMT": 
          typeName    =  "Iamt"
        elif typePred == "DIAM":
          typeName    =  "Diameter"
        EstimFolder = os.path.join(self.testSavePath,"Estimation-"+typeName,"Patient_"+str(p))
        self.makeFolder(EstimFolder)
        EstimFile   = os.path.join(EstimFolder,"Estim.jpg")
        self.saveValidLoss(gtVal,os.path.join(EstimFolder,  typeName+"GT.txt"))
        self.saveValidLoss(predVal,os.path.join(EstimFolder,typeName+"Pred.txt"))
        fig         = plt.figure()
        ax          = fig.add_subplot(111)
        ax.set_title(typeName+" Estimation of " + self.ModelName)
        gtVal       = self.cleanVect(gtVal)
        predVal     = self.cleanVect(predVal)
        ax.plot(gtVal,  '--', label="Ground truth "+typeName,color='r')
        ax.plot(predVal,'--', label="Prediction "+typeName,color='b')
        ax.set_ylabel(typeName+' [mm]')
        ax.set_xlabel('Time  [s]')
        ax.legend(loc='lower right')
        fig.savefig(EstimFile)
    
    def getCUDAVariable(self,data):
        dataVariable        =  Variable(data).cuda()
        return dataVariable
   
    def getCUDATargetsVariable(self,IamtLabels,DiamLabels):
        VariableIamtLabels  =  Variable(IamtLabels).cuda()
        VariableDiamLabels  =  Variable(DiamLabels).cuda()
        return VariableIamtLabels,\
               VariableDiamLabels

    def getValid(self,validListIMTVal,\
                 validListDiamVal,epochList):
        self.model.eval()
        LossvalidIMTVal               = []
        LossvalidDiamVal              = [] 
        validIMTVal                   = 0
        validDiamVal                  = 0
        contIter                      = 0
        loadData,loadLabels           =  self.validData.getTest() 
        numIter                       = loadData.shape[0]
        print("\n ==> Validation of ["+ str(numIter) +"] patients ..." )
        ValidProg = self.setProgressbar("Valid ",numIter*loadData.shape[1])
        for IterPatient in xrange(numIter):
            for IterSeq in xrange(loadData.shape[1]): 
                validLoadData          = None
                validLoadLabel         = None
                if self.frameSecond    != 0:
                    validLoadData      =  np.asarray(loadData  [IterPatient][IterSeq],dtype=np.float32)
                    validLoadLabel     =  np.asarray(loadLabels[IterPatient][IterSeq],dtype=np.float32) 
                else:
                    validLoadData      =  np.asarray(loadData  [IterPatient],dtype=np.float32)
                    validLoadLabel     =  np.asarray(loadLabels[IterPatient],dtype=np.float32)    
                getPredLabel           =  self.model(Variable(torch.from_numpy(validLoadData)).cuda())
                #IamtOutModel,\
                #DiamOutModel           =  self.formatValues(getPredLabel)
                IamtTarget,\
                DiamTarget             =  self.getDiamAndImT(validLoadLabel)
                varIamtTarget,\
                vatDiamTarget          =  self.getCUDATargetsVariable(torch.from_numpy(np.asarray(IamtTarget)),\
                                          torch.from_numpy(np.asarray(DiamTarget)))
                #validLoadData         =  self.getCUDAVariable(torch.from_numpy(validLoadData))
                if   self.typeMeasure == "Iamt":
                    validIMTVal        =  self.MSEcritEval(getPredLabel,varIamtTarget).data[0]  
                    LossvalidIMTVal.append  (validIMTVal)
                elif self.typeMeasure ==  "Diam":
                    validDiamVal       =  self.MSEcritEval(getPredLabel,vatDiamTarget).data[0]       
                    LossvalidDiamVal.append  (validDiamVal)
                contIter               += 1 
                ValidProg.update(contIter)
        ValidProg.finish()
        fig                            = plt.figure()
        ax                             = fig.add_subplot(111)
        ax.set_title("Validation of "  + self.ModelName)
        if  self.typeMeasure == "Iamt":
            meanIMTVal                     = np.mean(LossvalidIMTVal)
            validListIMTVal.append (meanIMTVal)
            validListIMTValPlot            = self.cleanVect(validListIMTVal)
            self.saveValidLoss(validListIMTVal, os.path.join (self.validSavePath, "ValidLossIMT.txt"))
            ax.plot(validListIMTValPlot,  '--', label="Validation IMT" ,color='b')
        elif self.typeMeasure ==  "Diam":     
            meanDiamVal                    = np.mean(LossvalidDiamVal)
            validListDiamVal.append(meanDiamVal)
            validListDiamValPlot           = self.cleanVect(validListDiamVal)
            self.saveValidLoss(validListDiamVal,os.path.join(self.validSavePath,  "ValidLossDiam.txt"))
            ax.plot(validListDiamValPlot, '--', label="Validation Diam",color='r')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mean Square Error (MSE)')
        ax.legend();
        fig.savefig(os.path.join(self.validSavePath,"ValidLoss.jpg"))
        self.model.train()
        return validListIMTVal,validListDiamVal
    
    def getTestAllSeqs(self,bestEpoch):
        test_loss    = 0 
        contIter     = 0
        lossIMTVal   = []
        lossDiamVal  = []
        testListIMT  = []
        testListDiam = []      
        typePred     = self.typeMeasure  
        print("\n ==> Best Epoch at: ["+ str(bestEpoch) +"]")
        self.saveBestModel(bestEpoch,typePred)
        bestModel                         = torch.load (os.path.join(self.modelBestPath,\
                                            "epoch_"+str(bestEpoch)+"_"+typePred+'.pt'))
        bestModel.eval()
        loadData,loadLabels               =  self.testData.getTest() 
        numIter                           =  loadData.shape[0]        
        print("\n ==> Testing of ["+ str(numIter) +"] patients ..." ) 
        TestProg = self.setProgressbar("Test ",numIter*loadData.shape[1])
        for IterPatient in xrange(numIter):
                testLoadData               =  np.asarray(loadData  [IterPatient],dtype=np.float32)
                testLoadLabel              =  np.asarray(loadLabels[IterPatient],dtype=np.float32)
                for IterFrames in xrange(testLoadLabel.shape[0]):
                    gtDiamVal              = []
                    predDiamVal            = []
                    gtIMTVal               = []
                    predIMTVal             = []
                    getPredLabel           = bestModel(Variable(torch.from_numpy(testLoadData)).cuda())
                    #TestIamtOutModel,\
                    #TestDiamOutModel       = self.formatValues(getPredLabel)
                    TestIamtTarget,\
                    TestDiamTarget         =  self.getDiamAndImT(testLoadLabel)
                    TestvarIamtTarget,\
                    TestvatDiamTarget      =  self.getCUDATargetsVariable(torch.from_numpy(np.asarray(TestIamtTarget)),\
                                              torch.from_numpy(np.asarray(TestDiamTarget)))
                    if    typePred == "Iamt":
                          testIMTVal       =  self.MSEcritEval(getPredLabel[IterFrames],TestvarIamtTarget[IterFrames]).data[0] 
                          lossIMTVal.append (testIMTVal) 
                    elif  typePred == "Diam":
                          testDiamVal      =  self.MSEcritEval(getPredLabel[IterFrames],TestvatDiamTarget[IterFrames]).data[0]
                          lossDiamVal.append(testDiamVal)   
                    if    typePred == "Iamt":
                          gtIMTVal.append  (TestIamtTarget[IterFrames])
                          predIMTVal.append(getPredLabel[IterFrames].data[0])
                    elif  typePred == "Diam":
                          gtDiamVal.append  (TestDiamTarget[IterFrames])
                          predDiamVal.append(getPredLabel[IterFrames].data[0])
                if    typePred == "Iamt": 
                      self.getDiamEsimation(gtIMTVal,predIMTVal,typePred,IterPatient)
                elif  typePred == "Diam":
                      self.getDiamEsimation(gtDiamVal,predDiamVal,typePred,IterPatient)
                contIter  += 1 
                TestProg.update(contIter)
        if    typePred     == "Iamt":
              meanlossIMTVal  = np.mean(lossIMTVal)
              stdlossIMTVal   = np.std (lossIMTVal)
        elif  typePred     == "Diam":
              meanlossDiamVal = np.mean(lossDiamVal)
              stdlossDiamVal  = np.std (lossDiamVal)
        if    typePred     == "Iamt":
              testListIMT.append ([meanlossIMTVal,stdlossIMTVal])
        elif  typePred     == "Diam":    
              testListDiam.append([meanlossDiamVal,stdlossDiamVal])
        if    typePred     == "Iamt":
              self.saveTestLoss(testListIMT,os.path.join(self.testSavePath, "TestLossIMT.txt"))
        elif  typePred     == "Diam":  
              self.saveTestLoss(testListDiam,os.path.join(self.testSavePath,"TestLossDiam.txt"))
        TestProg.finish()

    def getTest(self,bestEpoch):
        test_loss    = 0 
        contIter     = 0
        lossIMTVal   = []
        lossDiamVal  = []
        testListIMT  = []
        testListDiam = []
        typePred     = self.typeMeasure  
        print("\n ==> Best Epoch at: ["+ str(bestEpoch) +"]")
        self.saveBestModel(bestEpoch,typePred)
        bestModel                         = torch.load (os.path.join(self.modelBestPath,\
                                            "epoch_"+str(bestEpoch)+"_"+typePred+'.pt'))
        bestModel.eval()
        loadData,loadLabels                =  self.testData.getTest() 
        numIter                            =  loadData.shape[0]
        print("\n ==> Testing of ["+ str(numIter) +"] patients ..." )      
        TestProg = self.setProgressbar("Test ",numIter*loadData.shape[1])
        for IterPatient in xrange(numIter):
            gtDiamVal     = []
            predDiamVal   = []
            gtIMTVal      = []
            predIMTVal    = []
            for IterSeq in xrange(loadData.shape[1]): 
                if self.frameSecond       != 0:
                    testLoadData           =  np.asarray(loadData  [IterPatient][IterSeq],dtype=np.float32)
                    testLoadLabel          =  np.asarray(loadLabels[IterPatient][IterSeq],dtype=np.float32)
                else:
                    testLoadData           =  np.asarray(loadData  [IterPatient],dtype=np.float32)
                    testLoadLabel          =  np.asarray(loadLabels[IterPatient],dtype=np.float32)
                for IterFrames in xrange(testLoadLabel.shape[0]):
                    getPredLabel           = bestModel(Variable(torch.from_numpy(testLoadData)).cuda())
                    TestIamtOutModel,\
                    TestDiamOutModel       = self.formatValues(getPredLabel)
                    TestIamtTarget,\
                    TestDiamTarget         =  self.getDiamAndImT(testLoadLabel)
                    TestvarIamtTarget,\
                    TestvatDiamTarget      =  self.getCUDATargetsVariable(torch.from_numpy(np.asarray(TestIamtTarget)),\
                                              torch.from_numpy(np.asarray(TestDiamTarget)))
                    if    typePred == "Iamt":
                          testIMTVal       =  self.MSEcritEval(TestIamtOutModel[IterFrames],TestvarIamtTarget[IterFrames]).data[0]  
                          lossIMTVal.append (testIMTVal)
                    elif  typePred == "Diam":
                          testDiamVal      =  self.MSEcritEval(TestDiamOutModel[IterFrames],TestvatDiamTarget[IterFrames]).data[0] 
                          lossDiamVal.append(testDiamVal) 
                    if    typePred == "Iamt":
                          gtIMTVal.append  (TestIamtTarget[IterFrames])
                          predIMTVal.append(TestIamtOutModel[IterFrames].data[0])
                    elif  typePred == "Diam":
                          gtDiamVal.append  (TestDiamTarget[IterFrames])
                          predDiamVal.append(TestDiamOutModel[IterFrames].data[0]) 
                if    typePred == "Iamt": 
                      self.getDiamEsimation(gtIMTVal,predIMTVal,typePred,IterPatient)
                elif  typePred == "Diam":
                      self.getDiamEsimation(gtDiamVal,predDiamVal,typePred,IterPatient)
            contIter  += 1 
            TestProg.update(contIter)
        if    typePred     == "Iamt":
              meanlossIMTVal  = np.mean(lossIMTVal)
              stdlossIMTVal   = np.std (lossIMTVal)
        elif  typePred     == "Diam":
              meanlossDiamVal = np.mean(lossDiamVal)
              stdlossDiamVal  = np.std (lossDiamVal)
        if    typePred     == "Iamt":
              testListIMT.append ([meanlossIMTVal,stdlossIMTVal])
        elif  typePred     == "Diam":    
              testListDiam.append([meanlossDiamVal,stdlossDiamVal])
        if    typePred     == "Iamt":
              self.saveTestLoss(testListIMT,os.path.join(self.testSavePath, "TestLossIMT.txt"))
        elif  typePred     == "Diam":  
              self.saveTestLoss(testListDiam,os.path.join(self.testSavePath,"TestLossDiam.txt"))
        TestProg.finish()

    def getTrain(self):
        list_loss        = []
        contEpochs       = []
        validListIMTVal  = []
        validListDiamVal = []
        typePred         = self.typeMeasure 
        cEpoch           = 0
        loadPeaks        = None
        for EpochIndex in xrange(self.NumEpochs):
            print("\n ==> Epoch number: " + "["+ str(EpochIndex+1) +"] \n")
            epoch_loss        = 0
            TrainProg         = self.setProgressbar("Train ",self.NumIterations)
            for IterIndex in xrange(self.NumIterations):
                dataToPass    = []
                loadData     ,loadTrgetDiam,\
                loadTrgetIamt,loadPeaksDiam,\
                loadPeaksIamt,TaverageDiam ,\
                TaverageIamt ,nTimeStepDiam,\
                nTimeStepIam  =  self.trainData.getTrain() 
                ################################
                dataToPass.append(loadData     )
                dataToPass.append(loadTrgetDiam)
                dataToPass.append(loadTrgetIamt)
                dataToPass.append(loadPeaksDiam)
                dataToPass.append(loadPeaksIamt)
                dataToPass.append(TaverageDiam )
                dataToPass.append(TaverageIamt )
                dataToPass.append(nTimeStepDiam)
                dataToPass.append(nTimeStepIam )
                ###############################
                epoch_loss    += self.getClassification(dataToPass).data[0]
                dataToPass    = []
                TrainProg.update(IterIndex)
            TrainProg.finish()
            cEpoch += 1
            contEpochs.append(cEpoch)
            LossResult     = epoch_loss/self.NumIterations 
            list_loss.append(LossResult)
            print("\n\n ==> Epoch Loss: " + str(epoch_loss))
            # Create train valid graph
            self.getMSECurve(list_loss,contEpochs,"Train")
            # Save model
            self.saveModel(cEpoch)
            # get Validation
            validListIMTVal,\
            validListDiamVal = self.getValid(validListIMTVal,\
                               validListDiamVal,contEpochs)
        if self.frameSecond  == 0:
          if    typePred     == "Iamt":
             minindexIMT  = self.getMinValid_IMT_MSE()
             self.getTestAllSeqs(minindexIMT)
          elif  typePred     == "Diam":
             minindexDiam = self.getMinValid_Diam_MSE()
             self.getTestAllSeqs(minindexDiam)
        else:
          if    typePred     == "Iamt":
             minindexIMT  = self.getMinValid_IMT_MSE()
             self.getTest(minindexIMT)
          elif  typePred     == "Diam":
             minindexDiam = self.getMinValid_Diam_MSE()
             self.getTest(minindexDiam)
        self.deleteModelFolder()
        print("\n ==> DONE!")






                









