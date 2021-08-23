from keras.layers.recurrent import SimpleRNN
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import random
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch

inputSize = 50

#generates a set of seeded random intergers  of a certain size within the range(inclusive)
def generateSet(size):
    return [random.randint(0,10) for i in range(size)]

#one hot encode sequence
def oneHotEncode(seq):
    outputseq = []
    for i in seq:
        temp = [0 for i in range(10)]
        temp[i-1] = 1
        outputseq.append(temp)
    return np.array(outputseq)
#decode one hot encode
def decodeOneHot(seq):
    return [np.argmax(i) for i in seq]

def generateData(batchsize):
    X = []
    y = []
    for i in range(batchsize):
        print(i)
        #generate sequence
        sequence = generateSet(inputSize+1)
        #one hot encode
        encoded = oneHotEncode(sequence)
        # convert to 3d for input
        encoded = encoded.reshape(inputSize+1, encoded.shape[1])

        X.append(encoded[0:-1])
        y.append(encoded[-1])
    
    return np.array(X), np.array(y)




# define model
model = Sequential()
model.add(LSTM(10, input_shape=(inputSize, 10)))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# fit model
X = []
y = []
modelAccuracy = []
for i in range(500):
 
    X, y = generateData(100)    

    print(X.shape)
    print(y.shape)
    history = model.fit(X, y, epochs=1, batch_size=100, verbose=2)
    modelAccuracy.append(history.history["accuracy"][0])
# evaluate
X, y = generateData(1)
yhat = model.predict(X)
print("Expected:  %s" % decodeOneHot(y))
print("Predicted: %s" % decodeOneHot(yhat))

print(history.history.keys())
plt.plot(modelAccuracy)
axes = plt.gca()
axes.set_ylim([0,1])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



