import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from generateData import generateSet, ohe
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
import os
import datetime
import IPython
import IPython.display
import numpy as np
import pandas as pd
import seaborn as sns


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#meta
setSize = 10
batch_size = 5
##



def buildrnn():
    model = keras.Sequential()
    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128, input_shape = (batch_size, setSize)))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(1))


    return model

def trainrnn():
    model = buildrnn()
    trainVals = []
    trainEncs = []
    #create one batch of data
    for i in range(batch_size):
        trainVal, trainEnc = generateSet(i, 11, (1, 10))
        trainVals.append(trainVal)
        trainEncs.append(trainEnc)

    trainX = trainEncs
    trainY = np.array([i[1:] for i in trainX])
    trainX = np.array([i[:-1] for i in trainX])
    #print(trainX.shape)
    #print(trainX)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))

    # print(trainVals)

    # print(trainY)
    #
    print(trainX.shape)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.fit(trainX,trainY, setSize)


trainrnn()