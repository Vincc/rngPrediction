
from generateData import generateSet, ohe

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

print("packages imported")

#meta
setSize = 10
batch_size = 5

##


def decode(seq):
    return [i.index(1)+1 for i in seq]
def buildrnn():
    model = keras.Sequential()
    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128, batch_input_shape = (batch_size, setSize, 10), stateful=True))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))


    return model

def generateXy():
    #quantitative input
    trainEncs = [] #one hot encoded
    #create training data
    for i in range(batch_size):
        trainEnc = generateSet(11, (1, 10))
        trainEncs.append(trainEnc)

    trainX = trainEncs
    trainY = np.array([i[1:] for i in trainX])
    trainX = np.array([i[:-1] for i in trainX])
    trainX = np.reshape(trainX, (batch_size, setSize, trainX.shape[2]))

    return trainX, trainY
    #no hc encoding
    # trainX = [generateSet(i,11,(1,10)) for i in range(batch_size)]
    # print(trainX)
    # trainY = np.array([i[1:] for i in trainX])
    # trainX = np.array([i[:-1] for i in trainX])
    # print(trainX.shape)
    # print(trainX)
    # trainX = np.reshape(trainX, (batch_size, setSize, 1))
    # trainX.shape
    #

def trainrnn():

    model = buildrnn()


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    trainX, trainY = generateXy()
    # print(trainX)
    # print("~~~")
    # print(trainY)
    model.fit(trainX,trainY, batch_size = batch_size)
    print("done")

def testmodel():
    pass

trainrnn()