from keras.layers.recurrent import SimpleRNN
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import random
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch
import LFSR

inputSize = 20 #length of preceeding sequence
genRange = 255 #range of numbers generated
seedVal = random.randint(0,4095) #start of seed value
batch_size = 300 #number of datasets within each epoch


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
        
        random.seed(seedVal)
        #generate sequence

        #mersene or lsfr
        #sequence = generateSet(inputSize+1)
        sequence = generateLFSR(inputSize+1)
        seedVal = random.randint(0,4095)
        
        #one hot encode
        encoded = oneHotEncode(sequence)
        # convert to 3d for input
        encoded = encoded.reshape(inputSize+1, encoded.shape[1])

        X.append(encoded[0:-1])
        y.append(encoded[-1])
    
    return np.array(X), np.array(y)

# define model
model = Sequential()
model.add(SimpleRNN(10, input_shape=(inputSize, genRange)))
model.add(Dense(genRange, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# fit model
X = []
y = []


readstep = 1

modelAccuracy = []
tempacc = []
for i in range(50):
    
    print(i)
    X, y = generateData(batch_size)    

    print(X.shape)
    print(y.shape)
    history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2)
    tempacc.append(history.history["accuracy"][0])
    if i%readstep == 0:
        modelAccuracy.append(sum(tempacc)/len(tempacc))
# evaluate
correct = 0
for i in range(1000): 
    
    X, y = generateData(1)
    yhat = model.predict(X)
    if decodeOneHot(yhat) == decodeOneHot(y):
        correct+=1
    # print("Expected:  %s" % decodeOneHot(y))
    # print("Predicted: %s" % decodeOneHot(yhat))
print(correct/1000)

print(history.history.keys())
plt.plot(modelAccuracy)
# print(modelAccuracy)
# axes = plt.gca()
# axes.set_ylim([0,1])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



