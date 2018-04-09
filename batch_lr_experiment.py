

# ---------gradual warm-up
# > (k * LR) after 5 epochs

# --------linear scale batch....
# something with
# ink you could do it by just saving the weights after an epoch and then starting again
# doing a loop over model.fit epoch-by-epoch with different batch_size, loading the weights from the previous epoch each time


# https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/


# CNN for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)




# load the dataset but only keep the top n words, zero the rest
top_words = 24000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)





# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))#max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))