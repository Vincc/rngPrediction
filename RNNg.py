
from generateData import generateSet, ohe

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

import pandas as pd

print("packages imported")

#meta
setSize = 150
batch_size = 3
iterations = 50
#####


def decode(seq):
    return [np.argmax(i) for i in seq]
def buildrnn():
    model = keras.Sequential()
    # Add a LSTM layer with 128 internal units.
    model.add(layers.SimpleRNN(128, batch_input_shape = (batch_size, 1, 10), stateful=True))

    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10, activation='softmax'))


    return model



def generateXy():
    #quantitative input
    #create training data
    trainEnc = generateSet(setSize+2, (1, 10))


    trainX = trainEnc
    df = pd.DataFrame(trainX)
    df = pd.concat([df.shift(4), df.shift(3), df.shift(2), df.shift(1), df], axis=1)
    values = df.values
    values = values[5:, :]
    # convert to 3d for input
    trainX = values.reshape(len(values), 5, 10)
    # drop last value from y
    trainY = trainEnc[4:-1, :]

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

def trainrnn():

    model = buildrnn()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    for i in range(iterations):
        trainX, trainY = generateXy()
        #print(decode(trainX))
        #print(decode(trainY))
        # print(trainX.shape)
        # print(trainY.shape)
        model.fit(trainX,trainY, batch_size = batch_size, epochs = 1, verbose = 2)

    print("done")
    return model
def testmodel(model):
    testX, testY = generateXy()
    print(testX.shape)
    print(decode(testX))

    ypred = model.predict(testX, batch_size=batch_size)
    print('Expected:  %s' % decode(testY))
    print('Predicted: %s' % decode(ypred))

model = trainrnn()
testmodel(model)


# cce = keras.losses.CategoricalCrossentropy()
# print(cce([[0, 1, 0], [0, 0, 1]], [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]).numpy())
