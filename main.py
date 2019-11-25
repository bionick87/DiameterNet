########################################################
## Nicolo Savioli, PhD student King's Collage London  ##
########################################################

import argparse
import os
from train import Train

parser = argparse.ArgumentParser()
parser.add_argument('--typeExtractor',       required=True,             help= 'alexnet   | densenet | inception | resnet | vgg')
parser.add_argument('--typeRecurrent',       required=True,             help= 'copyframe | unidir   | bidir')
parser.add_argument('--typeCriterion',       required=True,             help= 'MSE       | MSECyclic')
parser.add_argument('--dataRealRoot',        default=                         '/data/ns14/dataset/ultrasound/split/', help='path for dataset')
parser.add_argument('--dataSyntheticRoot',   default=                         '/data/ns14/dataset/ultrasound-sim', help='path for dataset')
parser.add_argument('--dataSave',            default=                         '/data/ns14/ultrasound-loss',   help='path for save results')
parser.add_argument('--frameSecond',         type=int,   default=0,     help= 'number of ultrasound frame per second: 0 | 25, if 0 all video frames/ms else 25 frames/ms; with MSEPeak and MSECyclic only 0 value. Also with 25 frames/s only MSE! with 0 (all seq) only AlexNet!')
parser.add_argument('--dampingValue',        type=int,   default=1e-6,  help= 'damping value for non-peak and downstream values in the peak case value for non-peak and downstream values in the peak Loss')
parser.add_argument('--imageSize',           type=int,   default=128,   help= 'the height / width of the input image to network')
parser.add_argument('--lr',                  type=float, default=1e-3,  help= 'learning rate, default=1e-3')
parser.add_argument('--inChannels',          type=int,   default=1,     help= 'number of input channel')
parser.add_argument('--numEpochs',           type=int,   default=30,    help= 'number of epochs')
parser.add_argument('--numClass',            type=int,   default=1,     help= 'number of class')
parser.add_argument('--typeMeasurement',     required=True,             help= 'Diam | Iamt')
parser.add_argument('--typeDataset',         required=True,             help= 'Real | Synthetic')
opt = parser.parse_args()

print("\n\n\n")
print(" ===========================")
print(" === DiameterNet v 1.1  === ")
print(" ===========================")
print("\n")
print("\n ==> Options: \n")
print(opt)
print("\n\n\n")

trian = Train(opt.numClass,opt.inChannels,opt.typeExtractor,\
              opt.typeRecurrent,opt.typeCriterion,opt.dampingValue,\
              opt.imageSize,opt.lr,opt.frameSecond,\
              opt.numEpochs,opt.dataRealRoot,opt.dataSyntheticRoot,\
              opt.dataSave,opt.typeMeasurement,opt.typeDataset) 
trian.getTrain()






