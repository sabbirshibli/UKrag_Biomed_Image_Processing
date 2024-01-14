# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:02:17 2024

@author: Sabbir Ahmed Sibli
"""

# Help from:
# 1. https://youtu.be/GAYJ81M58y8
# 2. 

from Unet_Simplified import unet_model
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import os
from PIL import Image

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.utils import normalize
from keras.optimizers import Adam
from keras.layers import Input
from keras.layers import BatchNormalization, Conv2D, Activation
from keras.layers import MaxPool2D, Conv2DTranspose, Concatenate
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Loading the dataset (Original and Masked)
dicom_path = 'D:/Course Materials [Erasmus MSc]/University of Kragujevac/Biomedical Image Processing/Assignments/Datasets/Spine_DICOM'
masked_path = 'D:/Course Materials [Erasmus MSc]/University of Kragujevac/Biomedical Image Processing/Assignments/Datasets/masked_spines'

# Reading original dicom slices
original_dataset = imageio.volread(dicom_path, format='dcm')


# Reading masked images
masks = []
for file in Path(masked_path).iterdir():
    if not file.is_file():
        continue
    masks.append(imageio.imread(file))

mask_array = np.array(masks)      # converting list into numpy array
mask_array = mask_array[::-1]

# Resize Image to 128x128
im_height = 128
im_width = 128

X = np.zeros((len(original_dataset), im_height, im_width, 1), dtype=np.float32)     # Create array of zeros for data
y = np.zeros((len(mask_array), im_height, im_width, 1), dtype=np.float32)           #Create array of zeros for masks
for i in range(0,len(original_dataset)):
    # Load original images
    img = original_dataset[i]
    x_img = resize(img, (im_height, im_width, 1), mode = 'constant', preserve_range = True)
    # Load masks
    mask = mask_array[i]
    mask = resize(mask, (im_height, im_width, 1), mode = 'constant', preserve_range = True)
    # Creating Normalized image (converting all pixel values between 0 and 1)
    X[i] = x_img/255.0
    y[i] = mask/255.0

# Split training and validation data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Sanity check, view few masks according to their corresponding dicom slices
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Dicom Slice')
plt.imshow(np.reshape(X_train[image_number], (128, 128)), cmap='gray')
plt.subplot(122)
plt.title('Corresponding Mask')
plt.imshow(np.reshape(y_train[image_number], (128, 128)), cmap='gray')
plt.show()

# Training dataset using UNET Model
input_img = (im_height, im_width, 1)
model = unet_model(input_img)
model.compile(optimizer=Adam(learning_rate=10e-3), loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()


# Set up model checkpoints to save the best model during training
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
# Train the model
results = model.fit(
    X_train, y_train,
    epochs = 20,  # Set the number of epochs according to the requirement
    batch_size = 64,  # Adjust batch size for performance
    callbacks = [checkpoint],  # Add callbacks like ModelCheckpoint, EarlyStopping, etc.
    validation_data = (X_test, y_test),
    verbose = 1  # Set verbosity level as needed, Here 1 is progress bar
)

# load the best model
model.load_weights('best_model.h5')

# Visualizing Learning Curve
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();

# Plotting the training and validation loss at each epoch
loss = results.history['loss']
val_loss = results.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the training and validation accuracy at each epoch
acc = results.history['accuracy']
val_acc = results.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Accuracy Matrix: Intersection over Union (Jaccard Index)
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

# Selecting Random Predicted mask with ground truth
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

# Plotting predicted mask on testing image and testing labels
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()













