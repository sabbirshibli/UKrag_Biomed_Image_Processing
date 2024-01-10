# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:42:35 2024

@author: Sabbir Ahmed Sibli
"""

from keras.models import Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, Input, concatenate

def unet_model(inputs,num_features):
    inputs = Input(shape=(300,300,3))
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)
    
    # Base Layer
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(pool4)
    
    # Decoder Layer
    up6 = Conv2D(512,2,activation='relu',padding='same')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512,3,activation='relu',padding='same')(merge6)
    conv6 = Conv2D(512,3,activation='relu',padding='same')(conv6)
    
    up7 = Conv2D(256,2,activation='relu',padding='same')(UpSampling2D(size=(2,2)(conv6)))
    merge7 = concatenate([conv3,up7], axis=3)
    conv7 = Conv2D(256,activation='relu',padding='same')(merge7)
    conv7 = Conv2D(256,activation='relu',padding='same')(conv7)
    
    up8 = Conv2D(128,2,activation='relu',padding='same')(UpSampling2D(size=(2,2)(conv7)))
    merge8 = concatenate([conv2,up8], axis=3)
    conv8 = Conv2D(128,activation='relu',padding='same')(merge8)
    conv8 = Conv2D(128,activation='relu',padding='same')(conv8)
    
    up9 = Conv2D(64,2,activation='relu',padding='same')(UpSampling2D(size=(2,2)(conv8)))
    merge9 = concatenate([conv1,up9], axis=3)
    conv9 = Conv2D(64,activation='relu',padding='same')(merge9)
    conv9 = Conv2D(64,activation='relu',padding='same')(conv9)
    
    # Output Layer
    conv10 = Conv2D(inputs,1,activation='sigmoid')(conv9)
    
    # Defining model input output
    model = Model(inputs=inputs, outputs=conv10)
    model.Compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

Model_Unet = unet_model([100,100], 300)                    # I am confused what will be the parameters are

# Printing the summary of the model
print(Model_Unet.summary())

# Evaluation Matrix (using IoU)
import numpy as np

def IoU_evaluation(X_test, y_test):
    y_pred=Model_Unet.predict(X_test)
    y_pred_thresholded = y_pred > 0.5
    intersection = np.logical_and(y_test, y_pred_thresholded)
    union = np.logical_or(y_test, y_pred_thresholded)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)


