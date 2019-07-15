# encoding:utf-8
'''
"This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to transfer the
text data to the word indices.
'''

from __future__ import print_function

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import keras.backend as K



import load_data


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.840B/'
MAX_SEQUENCE_LENGTH = 2500
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.5

# first, build index mapping words in the embeddings set
# to their embedding vector
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
# and then vectorize the text samples into a 2D integer tensor
print('Processing text dataset')

x_train1 = load_data.train_text  # list of train text samples
y_train1 = load_data.train_label_ID  # list of train label ids
x_test = load_data.test_text # list of test text samples
y_test = load_data.test_label_ID# list of test label ids

texts = []
texts.extend(x_train1)
texts.extend(x_test)


labels = []
labels.extend(y_train1)
labels.extend(y_test)

labels_index = {'false':1, 'ddi':2, 'advise':3, 'effect':4, 'mechanism':5}


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_train_samples = len(x_train1)

x_train = data[:num_train_samples]
y_train = labels[:num_train_samples]
x_val = data[num_train_samples:]
y_val = labels[num_train_samples:]


# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# print('Conv1 output shape:', x.shape)
x = MaxPooling1D(5)(x)
# print('Maxpooling1 output shape:', x.shape)
x = Conv1D(128, 5, activation='relu')(x)
# print('Conv2 output shape:', x.shape)
x = MaxPooling1D(5)(x)
# print('Maxpooling2 output shape:', x.shape)
x = Conv1D(128, 5, activation='relu')(x)
# print('Conv3 output shape:', x.shape)
x = MaxPooling1D(5)(x)
# print('Maxpooling3 output shape:', x.shape)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(5, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', f1_score, precision, recall])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=30,
          validation_data=(x_val, y_val))