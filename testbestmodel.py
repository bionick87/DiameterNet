########################################################
## Nicolo Savioli, PhD stuendet King's Collage London ##
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
#from grad_cam import GradCam
from torchvision import models, transforms
warnings.filterwarnings("ignore")

class Test():
    def __init__(self,pathRootModel,pathTestset,\
                 pathSave,typeModel,typeGRU,\
                 frameSecond):
        self.pathRootModel = pathRootModel
        self.pathTestset   = os.path.join(pathTestset,"test.h5")
        self.typeModel     = typeModel
        self.typeGRU       = typeGRU
        self.pathSave      = pathSave
        self.model,\
        self.savedir,\
        self.ModelTag      = self.getModel()
        self.testData      = loder(self.pathTestset,\
                                   frameSecond)
        self.criterion     = self.getCriterion () 
        # Exsecution:
        self.getTest()
    
    def setProgressbar(self,description,iteration):
        widgets = [description, pb.Percentage(), ' ',
                   pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        timer = pb.ProgressBar(widgets=widgets, maxval=iteration).start()
        return timer

    def makedir(self,directory):
       if not os.path.exists(directory):
           os.makedirs(directory)
    
    def getCriterion(self):
        loss = torch.nn.MSELoss(size_average=True)
        print("\n ==> Convert loss to CUDA ...")
        loss = loss.cuda()
        return loss

    def getModel(self):
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
        model = self.openModel(os.path.join(self.pathRootModel,\
                                       nameModel,\
                                       "model.pt"))
        modelDir = os.path.join(self.pathSave,nameModel)
        self.makedir(modelDir)
        print("\n ==> Save model in dir : " + str(modelDir))
        return model,modelDir,nameModel
    
    '''
    def get_weights():
        for module in self.model.named_modules():
            if module[0] == 'ConvGRU.Conv_ct':
                GRUfeatures =  module[1].weight.data
                print(GRUfeatures)
    '''

    def reshape(self,vect): 
    	vect  = np.reshape(vect,(vect.shape[0],\
                  vect.shape[1]*vect.shape[2]))
        return vect
    
    def saveTXT(self,lossList,lossSavePath):
        with open(lossSavePath, "wb") as f:
            for i in xrange(lossList.shape[0]):
                f.write(str(lossList[i]) +"\n")

    def cleanVect(self,data):
        listPat = []
        for p in xrange(data.shape[0]):  
            listSeq = []
            for i in xrange(data.shape[1]):
                if data[p][i] > 0.5:
                   listSeq.append(data[p][i])
            listPat.append(np.asarray(listSeq))
        return np.asarray(listPat)  
     
    # Plots:

    def getPlotPlotIMTDiam (self,IMTlist,Diamlist,\
                 labelYname,savePathFig,numPat):
        fig         = plt.figure()
        ax          = fig.add_subplot(111)
        ax.set_title (self.ModelTag+" of MSE "+"patient "+str(numPat))
        ax.plot      (IMTlist,  '-', label="IMT" ,color='b')
        ax.plot      (Diamlist, '-', label="Diam",color='r')
        minLen      =  min(len(IMTlist),len(Diamlist))
        plt.xlim(0, minLen)
        ax.set_xlabel('Cardiac time [s]')
        ax.set_ylabel(labelYname)
        ax.legend();
        fig.savefig(savePathFig)

    
    def getPlotPlotGTPred (self,Predlist,GTlist,\
                 labelYname,savePathFig,numPat):
        fig         = plt.figure()
        ax          = fig.add_subplot(111)
        ax.set_title (self.ModelTag +" GT vs Prediction "+"patient " + str(numPat))
        ax.plot      (GTlist,  '-', label="GT of "    + labelYname ,color='r')
        ax.plot      (Predlist, '-', label="Pred of "  + labelYname ,color='g')
        minLen      =  min(len(GTlist),len(Predlist))
        plt.xlim(0, minLen)
        ax.set_xlabel('Cardiac time [s]')
        ax.set_ylabel('Distance [mm]')
        ax.legend();
        fig.savefig(savePathFig)

   
    # Regression plots:
    
    def reshapeVectPredGT(self,Pred,GT):
        minLen   = min (len(Pred),len(GT))
        PredList = []
        GTlist   = []
        for i in xrange(int(minLen)):
             PredList.append(Pred[i])
             GTlist.append  (GT  [i])
        return np.asarray(PredList),\
               np.asarray(GTlist)          
            

    def getRegresionPlot(self,Predlist,GTlist,\
                    labelYname,savePathFig,numPat):
        fig         = plt.figure()
        ax          = fig.add_subplot(111)
        Predlist,\
        GTlist      = self.reshapeVectPredGT(Predlist,GTlist)
        ###################################
        fit         = np.polyfit(Predlist,GTlist,1)
        fit_fn      = np.poly1d(fit) 
        ###################################
        ax.set_title ("Regression analysis of "+ self.ModelTag +" "+"patient " + str(numPat))
        #ax.plot      (GTlist,  '-', label="GT of "    + labelYname ,color='r')
        #ax.plot      (Predlist, '-', label="Pred of "  + labelYname ,color='g')
        ax.plot(Predlist,GTlist, 'ro', Predlist, fit_fn(Predlist), '-k')
        ax.set_xlabel('Automatic [mm]')
        ax.set_ylabel('Manual [mm]')
        ax.legend();
        fig.savefig(savePathFig)

     

    def getTest(self):
        test_loss    = 0 
        contIter     = 0
        loadData,loadLabels =  self.testData.getTest() 
        print("\n ==> Testing of ["+ str(loadData.data.size()[0]) +"] patients ..." ) 
        numIter  = loadData.data.size()[0]    
        TestProg = self.setProgressbar("Test ",numIter*loadData.data.size()[1])
        # MSE error
        PatientlossDiamVal  = []
        PatientlossIMTVal   = []
        # Prediction net
        PatientDiamVal_pred = []
        PatientIMTVal_pred  = []
        # gt 
        PatientDiamVal_gt   = []
        PatientIMTVal_gt    = []
        for IterPatient in xrange(numIter):
            # MSE error
            SeqlossDiamVal  = []
            SeqlossIMTVal   = []
            # Prediction net
            SeqDiamVal_pred = []
            SeqIMTVal_pred  = []
            # gt
            SeqDiamVal_gt   = []
            SeqIMTVal_gt    = []
            for IterSeq in xrange(loadData.data.size()[1]): 
            	# MSE error
                lossDiamVal   = []
                lossIMTVal    = []
                # Prediction net
                DiamVal_pred  = []
                IMTVal_pred   = []
                # gt
                DiamVal_gt    = []
                IMTVal_gt     = []
                testLoadData  =  loadData  [IterPatient][IterSeq]
                testLoadLabel =  loadLabels[IterPatient][IterSeq]
                Pred          =  self.model(testLoadData)
                for IterFrames in xrange(testLoadLabel.data.size()[0]):
                    testIMTVal    =  self.criterion(Pred[IterFrames][0],\
                                     testLoadLabel[IterFrames][0]).data[0]   
                    testDiamVal   =  self.criterion(Pred[IterFrames][1],\
                                     testLoadLabel[IterFrames][1]).data[0]
                    # MSE error
                    lossIMTVal.append   (testIMTVal)
                    lossDiamVal.append  (testDiamVal)
                    # Prediction net
                    IMTVal_pred.append  (float(Pred[IterFrames][0].data.cpu().numpy()))
                    DiamVal_pred.append (float(Pred[IterFrames][1].data.cpu().numpy()))
                    # gt
                    IMTVal_gt.append    (float(testLoadLabel[IterFrames][0].data.cpu().numpy()))
                    DiamVal_gt.append   (float(testLoadLabel[IterFrames][1].data.cpu().numpy()))
                contIter += 1 
                TestProg.update(contIter)
                # MSE error
                SeqlossIMTVal.append    (np.asarray(lossIMTVal))
                SeqlossDiamVal.append   (np.asarray(lossDiamVal))
                # Prediction net
                SeqIMTVal_pred.append   (np.asarray(IMTVal_pred))
                SeqDiamVal_pred.append  (np.asarray(DiamVal_pred))
                # gt
                SeqIMTVal_gt.append     (np.asarray(IMTVal_gt))
                SeqDiamVal_gt.append    (np.asarray(DiamVal_gt))
            # MSE error 
            PatientlossIMTVal.append   (np.asarray(SeqlossIMTVal)) 
            PatientlossDiamVal.append  (np.asarray(SeqlossDiamVal))
            # Prediction net
            PatientIMTVal_pred.append  (np.asarray(SeqIMTVal_pred))
            PatientDiamVal_pred.append (np.asarray(SeqDiamVal_pred))
            # gt 
            PatientIMTVal_gt.append    (np.asarray(SeqIMTVal_gt))
            PatientDiamVal_gt.append   (np.asarray(SeqDiamVal_gt))
        TestProg.finish()
        # MSE error 
        vectPatientlossDiamVal  = self.cleanVect(self.reshape(np.asarray(PatientlossDiamVal  )))
        vectPatientlossIMTVal   = self.cleanVect(self.reshape(np.asarray(PatientlossIMTVal   )))
        # Prediction net
        vectPatientDiamVal_pred = self.cleanVect(self.reshape(np.asarray(PatientDiamVal_pred )))
        vectPatientIMTVal_pred  = self.cleanVect(self.reshape(np.asarray(PatientIMTVal_pred  )))
        # gt 
        vectPatientDiamVal_gt   = self.cleanVect(self.reshape(np.asarray(PatientDiamVal_gt   )))
        vectPatientIMTVal_gt    = self.cleanVect(self.reshape(np.asarray(PatientIMTVal_gt    )))

        ##### Save MSE in section of 20 cardicac phase on txt ####

        for npat in xrange(vectPatientlossDiamVal.shape[0]):
            rootPath = os.path.join(self.savedir,"patient_"+str(npat))
            self.makedir(rootPath)
            ########################################################################
            pathPatientlossDiamVal = os.path.join(rootPath,"Frames-MSE-Diam.txt" )
            pathPatientlossIMTVal  = os.path.join(rootPath,"Frames-MSE-aimt.txt" )
            pathPatientDiamValPred = os.path.join(rootPath,"Frames-Pred-Diam.txt")
            pathPatientIMTValPred  = os.path.join(rootPath,"Frames-Pred-aimt.txt")
            pathPatientDiamValGT   = os.path.join(rootPath,"Frames-gt-Diam.txt"  )
            pathPatientIMTValGT    = os.path.join(rootPath,"Frames-gt-aimt.txt"  )
            #########################################################################
            self.saveTXT(vectPatientlossDiamVal [npat], pathPatientlossDiamVal  ) 
            self.saveTXT(vectPatientlossIMTVal  [npat], pathPatientlossIMTVal   ) 
            #########################################################################
            self.saveTXT(vectPatientDiamVal_pred[npat], pathPatientDiamValPred  ) 
            self.saveTXT(vectPatientIMTVal_pred [npat], pathPatientIMTValPred   ) 
            #########################################################################
            self.saveTXT(vectPatientDiamVal_gt  [npat], pathPatientDiamValGT    )  
            self.saveTXT(vectPatientIMTVal_gt   [npat], pathPatientIMTValGT     ) 
            #########################################################################

            pathPatientlossDiamAimtValPlot   = os.path.join(rootPath,"Frames-MSE-plot.jpg")
            pathPatientIMTValpredgtPlot      = os.path.join(rootPath,"Frames-aIMT-plot.jpg")
            pathPatientDiamValpredgtPlot     = os.path.join(rootPath,"Frames-Diam-plot.jpg")
            
            pathPatientIMTValRegressionPlot  = os.path.join(rootPath,"Frames-Regression-aIMT-plot.jpg")
            pathPatientDiamValRegressionPlot = os.path.join(rootPath,"Frames-Regression-Diam-plot.jpg")
            

            # General plots:

            self.getPlotPlotIMTDiam(vectPatientlossIMTVal[npat],vectPatientlossDiamVal[npat],\
                                    "MSE",pathPatientlossDiamAimtValPlot,npat)

            self.getPlotPlotGTPred(vectPatientIMTVal_pred[npat],vectPatientIMTVal_gt[npat],\
                                    "aIMT",pathPatientIMTValpredgtPlot,npat)

            self.getPlotPlotGTPred(vectPatientDiamVal_pred[npat],vectPatientDiamVal_gt[npat],\
                                    "Diam",pathPatientDiamValpredgtPlot,npat)
            # Rgression plots:                   
             
            self.getRegresionPlot(vectPatientIMTVal_pred[npat],vectPatientIMTVal_gt[npat],\
                                    "aIMT",pathPatientIMTValRegressionPlot,npat)

            self.getRegresionPlot(vectPatientDiamVal_pred[npat],vectPatientDiamVal_gt[npat],\
                                    "Diam",pathPatientDiamValRegressionPlot,npat)
               

    def getGRUfeatures(self,x):
        GRUfeatures = self.model.ConvGRU(self.model.model(x))
        return GRUfeatures
    
    def getGRUfeatures(self, x):
        x                   = self.reizeFeatures(x)
        h_next              = None    
        list_status         = [] 
        for time in xrange(x.data.size()[0]):
            h_next          = self.model.ConvGRU(self.model.model(x[time], h_next))
            list_status.append(h_next.size(0))
        return list_status
    
    def openModel(self, pathModel):
        model        = torch.load(pathModel)
        model.eval()
        return model
 
