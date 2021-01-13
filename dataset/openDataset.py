########################################################
## Nicolo Savioli, PhD student King's Collage London ##
########################################################

import h5py, os
import numpy as np
import cv2 
from   matplotlib import pyplot as plt
import copy 
import csv
import math

def resize_img(image,size):
    return cv2.resize(image, (size, size), icdnterpolation = cv2.INTER_CUBIC)

def open_hdf5(filename):
     f         = h5py.File(filename, 'r')
     data      = np.asarray(f['data'])
     label     = np.asarray(f['label']).transpose ((0,2,1))
     period    = np.asarray(f['period']).transpose((0,2,1))
     return  data,label,period

def saveDatasetHDF5(filename,data,label):
    f    = h5py.File(filename, 'w')
    dt   = h5py.special_dtype(vlen=bytes)
    dset = f.create_dataset('data' , shape =data.shape,   dtype ='f4', data=data)
    dset = f.create_dataset('label' ,shape =label.shape,  dtype ='f4', data=label)
    f.close() 

if __name__ == "__main__":
    
    ROOT       = "/data/ns14/dataset/ultrasound-sim"
    
    # old paths 
    trianPath        = os.path.join(ROOT,"train.h5")
    testPath         = os.path.join(ROOT,"test.h5")
    validPath        = os.path.join(ROOT,"valid.h5")

    TrainData,TrainLabel,TrainPeriod = open_hdf5(trianPath)

    print(TrainPeriod.shape)