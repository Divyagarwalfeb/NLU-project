import numpy as np
from pickle import dump
import string
import math
from random import randint
from pickle import load
from keras.models import load_model
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,CuDNNLSTM
from keras.layers import Embedding
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def lstm(X, Y, vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_length - 1))
    model.add(CuDNNLSTM(50))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=50, verbose=1)
    return (model)

def preprocessing_tokens(data,tokens):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    encoded = tokenizer.texts_to_sequences([data])[0]
    vocab_size = len(tokenizer.word_index)+1
    #encode 50 words -> 1 word
    in_size = 50
    sequences = list()
    for i in range(in_size, len(encoded)):
        sequence = encoded[i-in_size:i+1]
        sequences.append(sequence)
    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    sequences = array(sequences)
    X,Y = sequences[:,:-1],sequences[:,-1]
    return (tokenizer,X,Y, vocab_size, max_length)

#Read File
file = open("./NLU-project/austen-emma-persuasion-sense.txt", 'r')
data = file.read()
file.close()

#Token Generation
data = data.replace('--', ' ')
tokens = data.split()
table = str.maketrans('', '', string.punctuation)
tokens = [w.translate(table) for w in tokens]
tokens = [word for word in tokens if word.isalpha()]
tokens = [word.lower() for word in tokens]
index = int(len(tokens)*0.8)+1
train_tokens = tokens[0:index]
test_tokens = tokens[index:len(tokens)]
tokenizer,X_train,y_train, vocab_size, max_length = preprocessing_tokens(data,train_tokens)
Y_train = to_categorical(Y_train, num_classes=vocab_size)

#Training LSTM
model = lstm(X_train,y_train, vocab_size, max_length)

#Saving model
model.save('./NLU-project/model_word.h5')
dump(tokenizer,open('./NLU-project/tokenizer.pkl','wb'))

#loading model
model = load_model('./NLU-project/model_word.h5')
tokenizer = load(open('./NLU-project/tokenizer.pkl', 'rb'))

#Testing
tokenizer_test,X_test,Y_test, vocab_size_test, max_length_test = preprocessing(data,test_tokens)
Y_test = to_categorical(Y_test, num_classes=vocab_size)
loss = model.evaluate(X_test, Y_test, verbose=1)
print ("loss: ",loss[0])
perplexity = (2**loss[0])
print ("perplexity: ",perplexity)



