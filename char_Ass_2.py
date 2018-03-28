import numpy
import sys
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import to_categorical

def sampling(n_chars, n_voc):
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text_train[i:i + seq_length]
        seq_out = raw_text_train[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    Y = to_categorical(dataY, num_classes=n_voc)
    return (X, Y, dataX, dataY)

def lstm_model_generation(X, Y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=50, batch_size=128,verbose=1)
    return (model)


#Read file
filename = "austen-sense.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = raw_text.replace('--', ' ')
table = str.maketrans('', '', string.punctuation)
raw_text = [w.translate(table) for w in raw_text]

#sampleing
index = int(0.8*len(raw_text))
raw_text_train = raw_text[0:index]
raw_text_test = raw_text[index:len(raw_text)]
chars = sorted(list(set(raw_text_train)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text_train)
n_vocab = len(chars)
X,Y,dataX,dataY = sampling(n_chars,n_vocab)

#Saving the model
model = lstm_model_generation(X,Y)
model.save('model_char.h5')

#loading the model
model = load_model('model_char.h5')

#testing the model
chars_test = sorted(list(set(raw_text_test)))
n_chars_test = len(raw_text_test)
n_vocab_test = len(chars_test)
seq_length = 80
dataX_test = []
dataY_test = []
char_to_int_test = dict((c, i) for i, c in enumerate(chars_test))
int_to_char_test = dict((i, c) for i, c in enumerate(chars_test))
for i in range(0, n_chars_test - seq_length, 1):
    seq_in_test = raw_text_test[i:i + seq_length]
    seq_out_test = raw_text_test[i + seq_length]
    dataX_test.append([char_to_int_test[char] for char in seq_in_test])
    dataY_test.append(char_to_int_test[seq_out_test])
n_patterns_test = len(dataX_test)
X_test = numpy.reshape(dataX_test, (n_patterns_test, seq_length, 1))
Y_test = to_categorical(dataY_test, num_classes=n_vocab)

#calculating loss and perplexity
loss = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)
perplexity = 2**loss[0]
print ("loss: ",loss[0])
print ("perplexity: ",perplexity)










