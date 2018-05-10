import keras
import gensim
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


file = open('input.txt', encoding='utf-8').read()
lines = file.split('\n')
words = []
for line in lines:
    words.append(line.split(' '))

gmodel = gensim.models.Word2Vec(words, size=32, window=5, min_count=1, workers=4)

print ("vector len: " , gmodel['الماء'].__len__())
print (gmodel['كوب'].__len__())
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

train = (x_train, y_train)
test = (x_test, y_test)

max_review_length = 200
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)

print (x_train.__len__())

model = Sequential()
model.add(Dense(101, input_shape=(200,32)))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

