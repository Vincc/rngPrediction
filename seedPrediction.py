from keras.layers.recurrent import SimpleRNN
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import random
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch
import LFSR

inputSize = 100 #length of preceeding sequence
genRange = 255 #range of numbers generated
seedVal = np.random.randint(0, high = 4096) #start of seed value
batch_size = 30 #number of datasets within each epoch
trainingIterations = 500 #number of training iterations

#generates a set of seeded random intergers  of a certain size within the range(inclusive)
def generateSet(size):
    return [random.randint(0,genRange) for i in range(size)]

def generateLFSR(size):
    global seedVal
    seed = [0 for i in range(11)]
    seedBin = list(map(int, list(bin(seedVal)[2::])))
    seed[-len(seedBin)::] = seedBin
    register = LFSR.LFSR(fill=seed, taps=[8])
    return [register.rand(8) for i in range(size)]

#one hot encode sequence
def oneHotEncode(seq):
    outputseq = []
    for i in seq:
        temp = [0 for i in range(genRange)]
        temp[i-1] = 1
        outputseq.append(temp)
    return np.array(outputseq)
#decode one hot encode
def decodeOneHot(seq):
    return [np.argmax(i) for i in seq]

def generateData(batchsize):
    global seedVal
    X = []
    y = []
    for i in range(batchsize):
        
        
        #generate sequence

        #mersene or lsfr
        random.seed(seedVal)
        sequence = generateSet(inputSize+1)
        #sequence = generateLFSR(inputSize+1)
        
        seedVal = np.random.randint(0, high = 4096)
        
        #one hot encode
        encoded = oneHotEncode(sequence)
        # convert to 3d for input
        encoded = encoded.reshape(inputSize+1, encoded.shape[1])

        X.append(encoded[0:-1])
        y.append(encoded[-1])
    
    return np.array(X), np.array(y)

# define model
model = Sequential()
model.add(SimpleRNN(50, input_shape=(inputSize, genRange)))
model.add(Dense(genRange, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit model
X = []
y = []

readstep = 1

modelAccuracy = []
tempacc = []
for i in range(trainingIterations):
    
    print(i)
    X, y = generateData(batch_size)    

    print(X.shape)
    print(y.shape)
    history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2)
    
    
    modelAccuracy.append(history.history["accuracy"][0])
# evaluate
correct = 0
for i in range(100): 
    
    X, y = generateData(1)
    yhat = model.predict(X)
    if decodeOneHot(yhat) == decodeOneHot(y):
        correct+=1
    print("Expected:  %s" % decodeOneHot(y))
    print("Predicted: %s" % decodeOneHot(yhat))
    # if i %10 == 0:
    #     print(i)
print(correct/100)
print(modelAccuracy)




