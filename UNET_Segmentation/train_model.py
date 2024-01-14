# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:25:02 2024

@author: Sabbir Ahmed Sibli
source: https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
"""

from Unet_Portion import Unet
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# Set some params
im_width = 128
im_height = 128
border = 5

# Loading the dataset and the masks
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

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Sanity check, view few masks according to their corresponding dicom slices
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Dicom Slice')
plt.imshow(np.reshape(X_train[image_number], (128, 128)), cmap='gray')
plt.subplot(122)
plt.title('Corresponding Mask')
plt.imshow(np.reshape(y_train[image_number], (128, 128)), cmap='gray')
plt.show()


# Training dataset using Unet Model
input_img = Input((im_height, im_width, 1), name='img')
model = Unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Model Summary
model.summary()

callbacks = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

results = model.fit(X_train, y_train, batch_size=32, epochs=20, callbacks=callbacks, validation_data=(X_valid, y_valid))

# Visualizing

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
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Selecting Random Predicted mask with ground truth
test_img_number = random.randint(0, len(X_valid))
test_img = X_valid[test_img_number]
ground_truth=y_valid[test_img_number]
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

# Accuracy Matrix: Intersection over Union (Jaccard Index)
y_pred=model.predict(X_valid)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_valid, y_pred_thresholded)
union = np.logical_or(y_valid, y_pred_thresholded)

iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

# Accuracy Matrix: Dice Coefficien
dice_coefficient = (2.0 * np.sum(intersection)) / (np.sum(y_valid) + np.sum(y_pred_thresholded) + 1e-8)
print("Dice Coefficient:", dice_coefficient)

