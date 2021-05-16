import matplotlib.pyplot as plt
from generateData import generateSet
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#meta
trainSize = 10
##

trainXY = [generateSet(i, 10, (1,10)) for i in range(1,trainSize)]

def buildrnn():
    model = keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))

    model.summary()


def trainrnn():
    print(trainXY)

trainrnn()