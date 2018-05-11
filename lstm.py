import keras
import gensim
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import backend as K

file = open('input.txt', encoding='utf-8').read()
lines = file.split('\n')
words = []
for line in lines:
    words.append(line.split(' '))

gmodel = gensim.models.Word2Vec(words, size=32, window=5, min_count=1, workers=4)

x_train = []
x_test = []
c = 0
for line in lines:
    c+=1
    words = line.split(' ')
    wordvecs = []
    for word in words:
        wordvecs.append(gmodel[word])
    if (c <= 100):
        x_train.append(wordvecs)
    else:
        x_test.append(wordvecs)


y_train = []
y_test = []
yfile = open('classes.txt').read()
ylines = yfile.split('\n')
c = 0
for line in ylines:
    if (c < 100):
        y_train.append(int(line))
    else:
        y_test.append(int(line))
    c += 1


max_review_length = 200
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length) # pad sentences with zeros if less than 200 and truncate longer
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)


model = Sequential()
model.add(Dense(100, input_shape=(200,32))) # specifiying input shape (first param is op of layer)
model.add(LSTM(100)) # lstm rnn w 100 memory units
model.add(Dense(1, activation='sigmoid'))  # output layer - (classification)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=64)
# modifying epochs affects accurary, understand...

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# one way i found to print the testing result, but i doubted it
print(model.predict(x_test))

# searched for another one, got the same result
lstm_out = K.function([model.inputs[0],
                        K.learning_phase()],
                       [model.layers[2].output])

# pass in the input and set the the learning phase to 0
print(lstm_out([x_test, 0])[0])

# THEY ARE GIVING FLOATS MENNA!!!


