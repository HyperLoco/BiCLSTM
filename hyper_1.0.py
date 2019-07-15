'''
"This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to transfer the
text data to the word indices.
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Activation, concatenate, merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, Bidirectional, Dropout,BatchNormalization
from keras.models import Model
from keras.models import Sequential
import keras.backend as K
from tensorflow.contrib.metrics import streaming_precision



import load_data


def f1_score1(y_true, y_pred):
    num_tp = K.sum(y_true * y_pred)
    num_fn = K.sum(y_true * (1.0 - y_pred))
    num_fp = K.sum((1.0 - y_true) * y_pred)
    num_tn = K.sum((1.0 - y_true) * (1.0 - y_pred))
    f1 = 2.0 * num_tp / (2.0 * num_tp + num_fn + num_fp)
    return f1

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

def f2_score(y_true, y_pred, beta=2):
    true_positives = K.sum(y_true * y_pred)
    false_negatives = K.sum(y_true * (1.0 - y_pred))
    false_positives = K.sum((1.0 - y_true) * y_pred)
    true_negatives = K.sum((1.0 - y_true) * (1.0 - y_pred))
    precision = true_positives / (true_positives + ((1.0-y_true)*y_pred))
    recall = true_positives / (true_positives + false_negatives)
    f2 = (1+beta**2) * (precision * recall)/((beta**2*precision)+recall)
    return f2

# def precision_threshold(threshold=0.5):
#     def precision(y_true, y_pred):
#         """Precision metric.
#         Computes the precision over the whole batch using threshold_value.
#         """
#         threshold_value = threshold
#         # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#         y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#         # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#         true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
#         # count the predicted positives
#         predicted_positives = K.sum(y_pred)
#         # Get the precision ratio
#         precision_ratio = true_positives / (predicted_positives + K.epsilon())
#         return precision_ratio
#     return precision


# def recall_threshold(threshold = 0.5):
#     def recall(y_true, y_pred):
#         """Recall metric.
#         Computes the recall over the whole batch using threshold_value.
#         """
#         threshold_value = threshold
#         # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#         y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#         # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#         true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
#         # Compute the number of positive targets.
#         possible_positives = K.sum(K.clip(y_true, 0, 1))
#         recall_ratio = true_positives / (possible_positives + K.epsilon())
#         return recall_ratio
#     return recall


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
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
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

#embedding_layer = Embedding(num_words,
#                            EMBEDDING_DIM,
#                            input_length=MAX_SEQUENCE_LENGTH)

print('Training model.')

num_filter = 3
# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
embedded_sequences = BatchNormalization(epsilon=1e-08, mode=0, axis=1, momentum=0.9, weights=None)(embedded_sequences)
x1 = Conv1D(64, num_filter, activation='relu')(embedded_sequences)
x1 = Conv1D(128, num_filter, activation='relu')(x1)
x1 = Conv1D(256, num_filter, activation='relu')(x1)
x1 = Conv1D(512, num_filter, activation='relu')(x1)
x1 = MaxPooling1D(8)(x1)
x1 = LSTM(64, return_sequences=True)(x1)
#print('Conv4 output shape:', x.shape)

x2 = Conv1D(64, num_filter, activation='relu')(embedded_sequences)
x2 = Conv1D(128, num_filter, activation='relu')(x2)
x2 = Conv1D(256, num_filter, activation='relu')(x2)
x2 = Conv1D(512, num_filter, activation='relu')(x2)
x2 = MaxPooling1D(8)(x2)
x2 = LSTM(64, return_sequences=True)(x2)

#x = concatenate([x1, x2])
x = merge([x1, x2], mode='sum')

#print('Maxpooling4 output shape:', x.shape)
x = Dropout(0.7)(x)
x = Flatten()(x)
#x = Dense(2048, activation='relu')(x)
x = Dense(128, activation='relu')(x)
#x = Dense(512, activation='relu')(x)
preds = Dense(5, activation='softmax')(x)
model = Model(sequence_input, preds)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1_score, precision, recall, f2_score])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

# # train a Bidirectional LSTM model
# model = Sequential()
# model.add(Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='sigmoid'))