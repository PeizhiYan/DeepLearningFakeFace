# Deep Learning Face Face Generator
Author: Peizhi Yan 

Personal website: PeizhiYan.github.io

Affiliation: Lakehead University, Thunder Bay, Ontario, Canada

## Environment
### Software
Python3

Tensorflow-gpu

OpenCV

Matplotlib

Numpy

tqdm
### Hardware
Nvidia GTX 1080Ti

Xeon CPU


## Description
I trained a very deep convolutional autoencoder to reconstruct face image from the input face image. The input/output image size is 224x224x3, the encoded feature maps size is 7x7x64. Therefore, many less-important features will be ignored by the encoder (in other words, the decoder can only get limited information from the encoder). The training dataset has more than 200K celebrity images, the decoder will learn how to "draw" a face based on the encoded information (for instance: gender, hair color, etc.) and also to make a face looks like a celebrity.

## Files:
### dataset_helper.py
Load data.

### net3.py
My neural network architecture.

### train-net3.py
A jupyter notebook file, includes my training code, testing code (with result).

# For more information, please go to my homepage:
https://PeizhiYan.github.io
