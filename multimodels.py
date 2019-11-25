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

class thicknessnet(nn.Module):

        def __init__(self, num_classes=1, in_channels=1, typeCNN='A',\
                           typeGRU='bidir', sizeImage = 256, batch_norm=True):
            super(thicknessnet, self).__init__()
            self.num_classes  = num_classes
            self.batch_norm   = batch_norm
            self.in_channels  = in_channels
            self.ImageSize    = sizeImage
            self.model        = getCNNModle(typeCNN)
            self.typeGRU      = typeGRU 
            self.nPlans,\
            self.fsize        = self.getModelSize()
            if self.typeGRU == 'bidir':
               self.ConvGRU_1  = ConvGRUCell(self.nPlans,self.nPlans,3)
               self.ConvGRU_2  = ConvGRUCell(self.nPlans,self.nPlans,3)
               self.classifier = self.get_classifier(2)
            else: 
               self.ConvGRU    = ConvGRUCell(self.nPlans,self.nPlans,3)
               self.classifier = self.get_classifier(1)

        def getCNNModle(self,typeCNN):
            model = None
            if   typeCNN =="alexnet":
                model      = models.alexnet()
            elif typeCNN =="densenet":
                model = models.densenet161()
            elif typeCNN =="inception":
                model  = models.inception_v3()
            elif typeCNN =="resnet":
                model   = models.resnet18()
            elif typeCNN =="vgg":
                model = models.vgg16()
            return model
        
        def getSizeFeature(self,input):
            x           = self.layers(input)
            return x.data.size()[2]
        
        def getModelSize(self):
            output = self.model(slef.Input())
            nPlans = output.data.size()[1]
            fsize  = output.data.size()[2]
            return nPlans,fsize

        def reizeFeatures(self,x):
            x = x.view(x.data.size()[0],1,x.data.size()[1],\
                x.data.size()[2],x.data.size()[3])
            return x 
        

        def Input(self):
            image = Variable(torch.randn(1,1,self.ImageSize,self.ImageSize))
            return image  
        
        '''
         implement Bi-directionl GRUs 
        '''
        def bi_direction_GRU(self,x):
            x                    = self.reizeFeatures(x)
            outTensor            = Variable(torch.Tensor(x.data.size()[0],2)).cuda()
            h_next               = None     
            list_h_next_forward  = []
            list_h_next_backward = []
            list_concat          = []
            # Forward sequences 
            for time in xrange(x.data.size()[0]):
                h_next = self.ConvGRU_1(x[time], h_next)
                list_h_next_forward.append(h_next)
            # Backward sequences
            for time in reversed(range(0,x.data.size()[0])):
                h_next = self.ConvGRU_2(x[time], h_next)
                list_h_next_backward.append(h_next)
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
            outTensor           = Variable(torch.Tensor(x.data.size()[0],2)).cuda()
            for time in xrange(x.data.size()[0]):
                h_next          = self.ConvGRU(x[time], h_next)
                outTensor[time] = self.classifier(h_next.view(h_next.size(0), -1))[0]
            return outTensor

        def forward(self, input):
            x      = self.layers(input)
            output = None
            if self.typeGRU   == 'unidir':
                output = self.uni_direction_GRU   (x)
            elif self.typeGRU == 'bidir':
                output = self.bi_direction_GRU    (x)
            elif self.typeGRU == 'attention':
                output = self.attention_mechanism (x) 
            elif self.typeGRU == 'cnn':
                x      = x.view(x.size(0), -1)
                output = self.classifier(x)
            return output
            
        def get_classifier(self,num_GRUs):
            classifier = nn.Sequential(
                nn.Linear(self.nPlans*self.fsize*self.fsize,4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes))
            return classifier








