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

model = Sequential()
model.add(Conv2D(6,(5,5),activation='relu', input_shape=(28,28,3)))       # First Conv Layer (C1) with input 28x28
model.add(MaxPooling2D((2,2)))                                            # S2

model.add(Conv2D(16, (5,5), input_shape=(10,10,3)))                       # C3
model.add(MaxPooling2D((2,2)))                                            # S4

model.add(Conv2D(120, (5,5), input_shape=(10,10,3)))                       # C5


model.add(Flatten())
model.add(Dense(84,activation='relu'))                                    # F6 Fully Connected Layer
model.add(Dense(10,activation='softmax'))                                 # Output Layer with 10 Neurons


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # For Categorical Classification

print(model.summary())
