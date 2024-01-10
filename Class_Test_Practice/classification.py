# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:53:52 2024

@author: Sabbir Ahmed Sibli
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
 # Dense = Fully Connected Layer
 # Flatten = Convert Multidimention matrix to vector

'''
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('lables.csv', delimiter=',')

X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

X_train = X_train.reshape(len(X_train), 100,100,3)
Y_train = X_train.reshape(len(Y_train), 1)

X_test = X_train.reshape(len(X_test), 100,100,3)
Y_test = X_train.reshape(len(Y_test), 1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

X_train = X_train/255.0
X_test = X_test/255.0

idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()
'''

# Model Creation
'''
# Method 1
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # Output layer, if binary its sigmoid
    # Dense(5, activation='softmax') # multiclass its softmax
    ])
'''

# Method 2
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# Cost Function and Backpropagation
# lr = keras.optimizers.SGD(learning_rate=0.001) put it as and optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # For Binary Classification
# For categorical output loss='categorical_crossentropy'

'''
model.fit(X_train, Y_train, epochs=10, batch_size=64)

model.evaluate(X_test,Y_test)
'''
