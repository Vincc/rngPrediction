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
    pass


def trainrnn():
    print(trainXY)

trainrnn()