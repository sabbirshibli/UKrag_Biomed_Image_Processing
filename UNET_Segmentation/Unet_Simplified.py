# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:48:41 2024

@author: Sabbir Ahmed Sibli
"""
# Help: https://youtu.be/GAYJ81M58y8

# Importing Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import imageio
from pathlib import Path

from keras.models import Model
from keras.utils import normalize
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import BatchNormalization, Conv2D, Activation
from keras.layers import MaxPooling2D, Conv2DTranspose, Concatenate
from sklearn.preprocessing import MinMaxScaler

# Defining a separate function to do Convolutional Operations
def conv_block(input, num_filters):
    # First Conv Layer
    s = Conv2D(num_filters, 3, padding='same')(input)
    s = BatchNormalization()(s)  # For Normalization Purpose
    s = Activation('relu')(s)
    
    # Second Conv Layer
    s = Conv2D(num_filters, 3, padding='same')(s)
    s = BatchNormalization()(s)
    s = Activation('relu')(s)
    
    return s

# Defining a function for Entire Encoder Operation
# Encoder includes convolutional block and max pooling layer
def encoder_block(input, num_filters):
    s = conv_block(input, num_filters)
    p = MaxPooling2D((2,2))(s)
    return s, p

# Defining a function for Entire Decoder Operation
# Decoder includes an upsampling operation, a concatenation operation, convolutional block
def decoder_block(input, skip_features, num_filters):  # skip_featues are the output from the decoder block to be concatenated
    s = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    s = Concatenate()([s, skip_features])
    s = conv_block(s, num_filters)
    
    return s

def unet_model(input_shape):
    inputs = Input(input_shape)
    
    # Encoder Operation
    # First Layer
    s1,p1 = encoder_block(inputs, 64)
    # Second Layer
    s2,p2 = encoder_block(p1, 128)
    # 3rd Layer
    s3,p3 = encoder_block(p2, 256)
    # 4th Layer
    s4,p4 = encoder_block(p3, 512)
    
    # Base Layer
    b1 = conv_block(p4, 1024)
    
    # Decoder Operation
    # First Layer
    d1 = decoder_block(b1, s4, 512)
    # Second Layer
    d2 = decoder_block(d1, s3, 256)
    # 3rd Layer
    d3 = decoder_block(d2, s2, 128)
    # 4th Layer
    d4 = decoder_block(d3, s1, 64)
    
    # Output Layer
    outputs = Conv2D(1,1,padding='same',activation='sigmoid')(d4)
    
    model = Model(inputs, outputs, name='Unet-Model')
    return model

















