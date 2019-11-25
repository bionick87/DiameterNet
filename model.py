###############################################
## Nicolo Savioli, PhD King's Collage London ##
###############################################

import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import math
from   ConvGRUCell import ConvGRUCell
from torch.autograd import Variable
import numpy as np
import nets as models
import torch.nn.parallel

class thicknessnet(nn.Module):

        def __init__(self,num_classes,in_channels,typeCNN,\
                    typeGRU,sizeImage):
            super(thicknessnet, self).__init__()
            self.num_classes  = num_classes
            self.in_channels  = in_channels
            self.ImageSize    = sizeImage
            self.model        = self.getCNNModle(typeCNN)
            self.typeGRU      = typeGRU 
            self.nPlans,\
            self.fsize        = self.getModelSize()
            if self.typeGRU == 'bidir':
               print("\n ==> Bi-GRUs model activated.")
               self.ConvGRU_1  = ConvGRUCell(self.nPlans,self.nPlans,3)
               self.ConvGRU_2  = ConvGRUCell(self.nPlans,self.nPlans,3)
               self.classifier = self.get_classifier(2)
            else: 
               print("\n ==> GRU model activated.")
               self.ConvGRU    = ConvGRUCell(self.nPlans,self.nPlans,3)
               self.classifier = self.get_classifier(1)

        def getCNNModle(self,typeCNN):
            model = None
            if   typeCNN =="alexnet":
                print("\n ==> Features extractor: AlexNet.")
                model      = models.alexnet()
            elif typeCNN =="densenet":
                print("\n ==> Features extractor: Densenet 121.")
                model = models.densenet121()
            elif typeCNN =="inception":
                print("\n ==> Features extractor: Inception v4.")
                model  = models.inceptionv4()
            elif typeCNN =="resnet":
                print("\n ==> Features extractor: ResNet 18.")
                model   = models.resnet18()
            elif typeCNN =="vgg":
                print("\n ==> Features extractor: Vgg (E).")
                model = models.vgg16()
            return model
        
        def getSizeFeature(self,input):
            x           = self.layers(input)
            return x.data.size()[2]
        
        def getModelSize(self):
            output = self.model(self.getInput())
            nPlans = output.data.size()[1]
            fsize  = output.data.size()[2]
            return nPlans,fsize

        def reizeFeatures(self,x):
            x = x.view(x.data.size()[0],1,x.data.size()[1],\
                x.data.size()[2],x.data.size()[3])
            return x 
        
        def getInput(self):
            image = Variable(torch.randn(1,1,self.ImageSize,self.ImageSize))
            return image  
        
        '''
         implement Bi-directionl GRUs 
        '''
        def bi_direction_GRU(self,x):
            x                    = self.reizeFeatures(x)
            outTensor            = Variable(torch.Tensor(x.data.size()[0],self.num_classes)).cuda()
            h_next1              = None 
            h_next2              = None     
            list_h_next_forward  = []
            list_h_next_backward = []
            list_concat          = []
            # Forward sequences 
            for time in xrange(x.data.size()[0]):
                h_next1 = self.ConvGRU_1(x[time], h_next1)
                list_h_next_forward.append(h_next1)
            # Backward sequences
            for time in reversed(range(0,x.data.size()[0])):
                h_next2 = self.ConvGRU_2(x[time], h_next2)
                list_h_next_backward.append(h_next2)
            # Concatenation features
            for time in xrange(x.data.size()[0]):
                list_concat.append(torch.cat((list_h_next_forward[time],\
                list_h_next_backward[time]),1))
            # Classification 
            for time in xrange(x.data.size()[0]):
                outTensor[time] = self.classifier(list_concat[time].view(list_concat[time].size(0),-1))[0]
            return outTensor

        '''
        Implement Uni-directionl GRUs 
        '''
        def uni_direction_GRU(self, x):
            x                   = self.reizeFeatures(x)
            h_next              = None     
            list_classification = []
            outTensor           = Variable(torch.Tensor(x.data.size()[0],self.num_classes )).cuda()
            for time in xrange(x.data.size()[0]):
                h_next          = self.ConvGRU(x[time], h_next)
                outTensor[time] = self.classifier(h_next.view(h_next.size(0), -1))[0]
            return outTensor
        
        def getModel(self,input):
            x      = self.model(input)
            output = None
            if self.typeGRU   == 'unidir':
                output = self.uni_direction_GRU   (x)
            elif self.typeGRU == 'bidir':
                output = self.bi_direction_GRU    (x)
            elif self.typeGRU == 'attention':
                output = self.attention_mechanism (x) 
            elif self.typeGRU == 'copyframe':
                x      = x.view(x.size(0), -1)
                output = self.classifier(x)
            return output

        def forward(self, input):
            return  self.getModel(input)    

        def get_classifier(self,num_GRUs):
            classifier = nn.Sequential(
                nn.Linear(num_GRUs*self.nPlans*self.fsize*self.fsize,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes))
            return classifier








