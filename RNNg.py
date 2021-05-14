import matplotlib.pyplot as plt
from generateData import generateSet
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
#meta
trainSize = 10
##

trainXY = [generateSet(i, 10, (1,10)) for i in range(1,trainSize)]

def buildrnn():
    pass

def trainrnn():
    print(trainXY)

trainrnn()