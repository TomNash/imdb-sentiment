import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, Dense, GRU, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import pickle

#
model_file_path = 'models/imdb_sentiment_RNN.h5'
weights_file_path = "models/weights.best.hdf5"

# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)

# Distinct word count
print("Number of distinct words: %d" & (len(numpy.unique(numpy.hstack(X)))))

# Summarize review length
result = [len(x) for x in X]
print("Review length: \nMean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))

# plot review length
plt.boxplot(result)
plt.show()

top_words = 5000
seq_length = 500

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=seq_length)
X_test = sequence.pad_sequences(X_test, maxlen=seq_length)

checkpoint = ModelCheckpoint(weights_file_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='max')

# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=seq_length))
model.add(GRU(4))
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=100, verbose=1,
                    callbacks=[EarlyStopping(min_delta=0.001, patience=2), checkpoint])


# Final evaluation of the model
print("Accuracy: %.2f%%" % (model.evaluate(X_test, y_test, verbose=0)[1]*100))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model_json = model.to_json()
pickle.dump(model_json, open(model_file_path, 'wb'))
