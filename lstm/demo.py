from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras
import numpy as np
from keras.datasets import imdb
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D

# http://geek.csdn.net/news/detail/232869 教程地址

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train[0])
print(np.max(X_train[0]))


maxword = 400
X_train = sequence.pad_sequences(X_train, maxlen = maxword)
X_test = sequence.pad_sequences(X_test, maxlen = maxword)
vocab_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1
print(vocab_size)
print(vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length = maxword))
model.add(Conv1D(filters = 64, kernel_size = 3, padding = 'same', activation= 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.25))
model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'same',activation= 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics =['accuracy'])
print(model.summary())