# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:05:21 2019

@author: MUJ
"""

# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

#first Convolution layer
classifier.add(Conv2D(6, (5, 5), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

#second convolution layer
classifier.add(Conv2D(12, (5, 5), activation = 'relu'))
classifier.add(AveragePooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 192, activation = 'relu'))
classifier.add(Dense(units=4,activation='sigmoid'))


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('binarized dataset/training set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('binarized dataset/test set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch =300,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 180)

#single predicition

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('binarized dataset/binarized_prediction/c_or_palm2.png',
                            target_size = (128, 128,3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices  
