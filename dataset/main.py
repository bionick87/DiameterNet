########################################################
## Nicolo Savioli, PhD stuendet King's Collage London ##
########################################################

import h5py, os
import numpy as np
import cv2 
from   matplotlib import pyplot as plt
import copy 
import csv
import math

def show_image(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()     

def save_ultra_seq(data,path_save,num_pat):
  seq_path = os.path.join(path_save,"sequence_"+str(num_pat))
  if not os.path.exists(seq_path):
    os.makedirs(seq_path)
  for frame in xrange(data.shape[0]):
      cv2.imwrite(os.path.join(seq_path,"frame_" + str(frame)\
                                   + '.jpg'), data[frame][0])
def open_image(image):
    return cv2.imread(image,0)

def open_hdf5(filename):
     f         = h5py.File(filename, 'r')
     data      = np.asarray(f[f.keys()[0]]).transpose((2,3,1,0))
     return  data 

def split_dataset(data,labels,percent):
    train_data      = []
    valid_data      = []
    train_label     = []
    valid_label     = []
    permute_indices = np.random.permutation(data.shape[0])
    for i in range(data.shape[0]):
        if i > percent:
            valid_index    = permute_indices[i]
            valid_data.append (data  [valid_index])
            valid_label.append(labels[valid_index])
        else:
            train_index    = permute_indices[i]
            train_data.append (data  [train_index])
            train_label.append(labels[train_index])
    return  np.asarray(train_data), np.asarray(train_label),\
            np.asarray(valid_data), np.asarray(valid_label)

def read_csv(file):
  f = open(file, 'rt')
  read_list = []
  try:
    reader = csv.reader(f)
    for row in reader:
        list_value = []
        list       = row[0].strip().split(' ')
        for value in list:
            if value == '':
               continue
            get_float  = 0 
            if math.isnan(float(value))\
                or float(value)<0:
                get_float = 0.0
            else:
                get_float = float(value)
            list_value.append(get_float)
        read_list.append(list_value)
  finally:
    f.close()
  return read_list

def check_patient(data,aIMTVal_label,DiamVal_list):
    if data == aIMTVal_label== DiamVal_list:
       print("\n ==> Correct number of patients: "+"["+str(data)+"]"+"  --|-- [OK]")
    else:
       print("\n ==> ERROR: number of patients not correct!")

def resample_labels(label_seq,maxValue):
    new_seq = [] 
    for p_labels in label_seq:
        cont    = 0
        tmp_seq = []
        for values in p_labels:
            tmp_seq.append(values)  
            if cont > maxValue-2:
              break    
            cont = cont + 1  
        new_seq.append(tmp_seq)  
    return new_seq

def get_labels_array(list_label,label_container,\
                     num_p,num_frames,num_class):
    cont_p = 0 
    for plabel in list_label:
      cont_val = 0 
      for val in plabel:
        label_container[cont_p][num_class][cont_val] = val
        cont_val=cont_val+1
      cont_p=cont_p+1 
    return label_container

def resize_img(image,size):
    return cv2.resize(image, (size, size), interpolation = cv2.INTER_CUBIC)

def saveDatasetHDF5(filename,data,label):
    f    = h5py.File(filename, 'w')
    dt   = h5py.special_dtype(vlen=bytes)
    dset = f.create_dataset('data', shape =data.shape,  dtype ='f4', data=data)
    dset = f.create_dataset('label',shape=label.shape, dtype ='f4', data=label)
    f.close() 

def create_data(pathFolder):
    p_lists      = sorted(os.listdir(pathFolder)) 
    list_num_f   = []  
    for patient in p_lists:
        data     = open_hdf5(os.path.join(pathFolder,patient))
        list_num_f.append(data.shape[0])
    max_frame    = np.max(list_num_f)  
    zeros_mask   = np.zeros((data.shape[1],\
                    data.shape[2],data.shape[3]))
    acc_list     = [] 
    for patient in p_lists:
        print("\n ==> Read : " + patient)
        np_frames      = None
        data           = open_hdf5(os.path.join(pathFolder,patient))
        if data.shape[0] < max_frame:
          frames_list  = []
          for sr_copy in xrange(data.shape[0]):
              frames_list.append(data[sr_copy])
          for sr_copy in xrange(max_frame-data.shape[0]):
              frames_list.append(zeros_mask)
          np_frames    = np.asarray(frames_list)
        else:
          np_frames    = np.asarray(data)
        acc_list.append(np_frames) 
    return np.asarray(acc_list)

def make_dir(path):
    if not os.path.exists(path):
       os.makedirs(path)

if __name__ == "__main__":
    # ROOT PATH
    ROOT         = "/home/ns14/Desktop/workspace/Projects/dataset" 
    aIMTVal_path = os.path.join(ROOT,"aIMTVal.csv")
    DiamVal_path = os.path.join(ROOT,"DiamVal.csv")
    data_path    = os.path.join(ROOT,"Data")
    # open labels dataset 
    print("\n ==> Create Dataset ...")
    dataset      = create_data(data_path)
    # open .CSV files and resample its at mx num of frames 
    num_patients = dataset.shape[0]
    num_frames   = dataset.shape[1]
    aIMTVal_list = resample_labels(read_csv(aIMTVal_path),num_frames)
    DiamVal_list = resample_labels(read_csv(DiamVal_path),num_frames)
    # check correct num patients 
    check_patient(dataset.shape[0],\
                  len(aIMTVal_list),len(DiamVal_list))
    print("\n ==> Create labels Tensor...")
    allocate_labels          = np.zeros((num_patients,2,num_frames))
    allocalte_labels_1_class = get_labels_array(aIMTVal_list,allocate_labels,\
                                                   num_patients,num_frames,0)
    labels                   = get_labels_array(DiamVal_list,allocalte_labels_1_class,\
                                                   num_patients,num_frames,1)
    img_train,labels_train,\
    img_test,label_test   = split_dataset(dataset,labels,16)
    img_test_t,labels_test_t,\
    img_valid,label_valid = split_dataset(img_test,label_test,4)

    print("\n ==> Make save hdf5 dir...")
    save_hdf5_files = os.path.join(ROOT,"ultrasound_h5")
    make_dir(save_hdf5_files)

    print("\n ==> Save train ...")
    train_h5 = os.path.join(save_hdf5_files,"train.h5")
    saveDatasetHDF5(train_h5,img_train,labels_train)

    print("\n ==> Save test ...")
    test_h5 = os.path.join(save_hdf5_files,"test.h5")
    saveDatasetHDF5(test_h5,img_test,label_test)

    print("\n ==> Save valid ...")
    valid_h5 = os.path.join(save_hdf5_files,"valid.h5")
    saveDatasetHDF5(valid_h5,img_valid,label_valid)

    
   











