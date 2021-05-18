from random import randint
from numpy import array
from numpy import argmax
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# generate a sequence of random numbers in [0, 99]
def generate_sequence(length=25):
    return [randint(0, 99) for _ in range(length)]


# one hot encode sequence
def one_hot_encode(sequence, n_unique=100):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# generate data for the lstm
def generate_data():
    # generate sequence
    sequence = generate_sequence()
    # one hot encode
    encoded = one_hot_encode(sequence)
    # convert to 3d for input
    X = encoded.reshape(encoded.shape[0], 1, encoded.shape[1])
    return X, encoded


# define model
model = Sequential()
model.add(LSTM(15, input_shape=(1, 100)))
model.add(Dense(100, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
for i in range(500):
    X, y = generate_data()
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)
# evaluate model on new data
X, y = generate_data()
yhat = model.predict(X)
print('Expected:  %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))

