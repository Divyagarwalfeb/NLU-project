import string
from pickle import load
from random import randint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy import array

def token_generation(document):
    document = document.replace('--', ' ')
    tokens = document.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def preprocessing_tokens(tokens,data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    encoded = tokenizer.texts_to_sequences([data])[0]
    vocab_size = len(tokenizer.word_index)+1
    in_size = 2
    sequences = list()
    for i in range(in_size, len(encoded)):
        sequence = encoded[i-in_size:i+1]
        sequences.append(sequence)
    max_length = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
    sequences = array(sequences)
    X,Y = sequences[:,:-1],sequences[:,-1]
    return (tokenizer,X,Y, vocab_size, max_length)

def sentence_generation(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
    print(in_text)
    
#Read File
file = open("austen-emma_austen-persuasion_austen-sense.txt", 'r')
data = file.read()
file.close()

#Preprocessing
tokens = token_generation(data)
index = int(len(tokens)*0.8)+1
train_tokens = tokens[0:index]
test_tokens = tokens[index:len(tokens)]
tokenizer,X_train, Y_train, vocab_size, max_length = preprocessing_tokens(train_tokens,data)
Y_train = to_categorical(Y_train, num_classes=vocab_size)

#Load Model
model = load_model('model3_word.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length=2;

#Sentence generation
temp = randint(0,len(train_tokens)-max_length)
seed_text = train_tokens[temp]
for i in range(1,max_length-1):
    temp = temp+1
    seed_text += ' ' + train_tokens[temp]
n_words = 10
sentence_generation(model, tokenizer, max_length-1, seed_text, n_words)