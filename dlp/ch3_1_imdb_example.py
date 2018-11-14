import numpy as np
import keras as ks
from keras.datasets import imdb
from keras import models
from keras import layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_seqs(seqs, dim=10000):
    results = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results

x_train = vectorize_seqs(train_data)
x_test = vectorize_seqs(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]
p_x_train = x_train[10000:]

y_val = y_train[:10000]
p_y_train = y_train[10000:]

history = model.fit(p_x_train, p_y_train, epochs=20,batch_size=512,validation_data=(x_val, y_val))

history_dict = history.history
