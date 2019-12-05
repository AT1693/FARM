# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:29:48 2018

@author: mohit123
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(8, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Flatten())

#hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(p = 0.5))

#output layer
classifier.add(Dense(output_dim = 4, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(128, 128),
        batch_size=64,
        class_mode='categorical' )
label_map = (training_set.class_indices)

print(label_map)

itr = test_set = test_datagen.flow_from_directory(
        'testing/test1',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('keras_grapes_trained_model_weights_2000.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    if(arr[i] == 0):
        j += 1
    i += 1

print ('Correct Early Blight predictions')
print(j/len(arr))

# Healthy test cases
itr = test_set = test_datagen.flow_from_directory(
        'testing/test2',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('keras_grapes_trained_model_weights_2000.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    if(arr[i] == 2):
        j += 1
    i += 1

print ('Correct healthy predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'testing/test3',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('keras_grapes_trained_model_weights_2000.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    if(arr[i] == 1):
        j += 1
    i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))

itr = test_set = test_datagen.flow_from_directory(
        'testing/test4',
        target_size=(128, 128),
        batch_size=377,
        class_mode='categorical')

X, y = itr.next()

classifier.load_weights('keras_grapes_trained_model_weights_2000.h5')

#for layer in classifier.layers:
#    g=layer.get_config()
#    h=layer.get_weights()
#    print (g)
#    print (h)

#scores = classifier.evaluate_generator(test_set,62/32)
arr = classifier.predict_classes(X, batch_size=377, verbose=1)
i = 0
j = 0
print(arr)
while i < len(arr):
    if(arr[i] == 1):
        j += 1
    i += 1

print ('Correct Late blight predictions:')
print(j/len(arr))


#print(j)
    
#print(arr)
#print(len(scores))
#print(scores)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])