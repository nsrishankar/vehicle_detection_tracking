# Functions that comprise of all decided classifiers, as well as some deep learning algorithms
# Implementing real-time YOLO and tiny_YOLO networks in Keras using weights and architecture specified in:
# You Only Look Once: Unified, Real-Time Object Detection [Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi] CVPR2016

import numpy as np
import cv2
import pickle
import time
import keras 

from keras.layers.core import Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU

# Build a tiny_YOLO model (smaller model than YOLO, but less accurate)
def build_tinyyolo(): 

    model=Sequential()
    model.add(Convolution2D(16,3,3,input_shape=(3,448,448),init='normal',border_mode='same',subsample=(1,1),name='Conv1'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool1'))

    model.add(Convolution2D(32,3,3,border_mode='same',name='Conv2'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool2'))

    model.add(Convolution2D(64,3,3,border_mode='same',name='Conv3'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool3'))

    model.add(Convolution2D(128,3,3,border_mode='same',name='Conv4'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool4')) 

    model.add(Convolution2D(256,3,3,border_mode='same',name='Conv5'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool5'))

    model.add(Convolution2D(512,3,3,border_mode='same',name='Conv6'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2,2),name='MaxPool6'))                       

    model.add(Convolution2D(1024,3,3,border_mode='same',name='Conv7'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Convolution2D(1024,3,3,border_mode='same',name='Conv8'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Convolution2D(1024,3,3,border_mode='same',name='Conv9'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1470))

if __name__=="__main__":
#Build & compile the model
    tiny_YOLO=build_tinyyolo()
    print(tiny_YOLO.summary())