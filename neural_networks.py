# Functions that comprise of all decided classifiers, as well as some deep learning algorithms

import numpy as np
import cv2
import pickle
import time

from keras.layers.core import Dense, Lambda, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping


# Build a LeNet-5 basic
def LeNet5():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, name = 'Lambda', input_shape=(32, 32, 3)))
    model.add(Convolution2D(6, 5, 5, init = 'normal', activation = 'elu', border_mode = 'valid', subsample = (1,1), 
                            bias=True, name = 'Conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool1'))
    model.add(Convolution2D(16, 5, 5, init = 'normal', activation = 'elu', border_mode = 'valid', subsample = (1,1), 
                            bias=True, name = 'Conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name = 'MaxPool2'))
    model.add(Flatten(name = 'Flatten'))
    model.add(Dense(120, activation = 'elu', name = 'FC1'))
    model.add(Dropout(keep_prob, name = 'Dropout1'))
    model.add(Dense(84, activation = 'elu', name = 'FC2'))
    model.add(Dropout(0.5, name = 'Dropout2'))
    #Sigmoid activation function is used to get a probability output
    model.add(Dense(1, activation = 'sigmoid', name = 'Output'))
    model.compile(optimizer = 'adam', loss =  'binary_crossentropy', metric = ['accuracy'])
    
    return model

#Build & compile the model
LeNet5 = build_LeNet5(keep_prob)
print(LeNet5.summary())