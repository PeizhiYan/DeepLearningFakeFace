##################################################
# Author: Peizhi Yan                             #
# Affiliation: Lakehead University               #
# Personal Website: https://PeizhiYan.github.io  #
# Date: Jan. 21, 2019                            #
##################################################

import os
import cv2
import numpy as np

"""get all the image names under a given directory"""
def get_names(path):
    ret = []
    for name in os.listdir(path):
        if not name.startswith('_') and name.endswith('.jpg'):
            ret.append(name)
    return np.array(ret)

"""load and reshape image"""
def load_img(path, f_name):
    img = cv2.imread(path+f_name) # load image via OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB
    img = img/255 # normalize the pixel range from 0~255 to 0.0~1.0
    return cv2.resize(img, (224, 224)) # resize to 224x224x3

"""load dataset"""
def load_dataset(path, n):
    '''n is the number of image you want to load, if n == -1, load all'''
    list_names = get_names(path)
    
    '''random permutate the list of names'''
    p = np.random.permutation(len(list_names))
    list_names = list_names[p]
    
    dataset = []
    if n == 0: return
    if n == -1:
        for name in list_names:
            dataset.append(load_img(path, name))
    else:
        for i in range(n):
            dataset.append(load_img(path, list_names[i]))
    dataset = np.array(dataset)
    return dataset

"""image normalization"""
'''
def norm(img):
    R_mix 
'''