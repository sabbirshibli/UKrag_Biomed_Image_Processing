# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:44:25 2024

@author: Sabbir Ahmed Sibli
"""
# Source: https://towardsdatascience.com/easy-method-of-edge-detection-in-opencv-python-db26972deb2d
# Importing Important Libraries
from pathlib import Path
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dicom2jpg
from PIL import Image

# Canny Edge Detection
# Step 01: Reducing noise using the Gaussian smoothing
# Step 02: Computing the gradients
# Step 03: Applying non-maxima suppression to reduce the noise and have only the local maxima in the direction of the gradient
# Step 04: Finding the upper and lower threshold
# Step 05: Applying the threshold

# Reading Image
img = cv2.imread('image.jpg')
image = img[:, 50:420]  # Selecting SubImage on ROI

# Applying Gaussian Smoothing/Blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7,7), 0)

# Visualizing Blurred Image
plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blurred,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Applying Canny Edge Detection Algorithm
# cv2.Canny(image, lower_threshold, upper_threshold)

# Applying Canny in three different thresholding ranges (Trial & Error Method)
wide = cv2.Canny(blurred, 50, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 210, 250)

# Visualizing Edged Images
plt.subplot(131),plt.imshow(wide,cmap = 'gray')
plt.title('Wide'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mid,cmap = 'gray')
plt.title('Mid'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(tight,cmap = 'gray')
plt.title('Tight'), plt.xticks([]), plt.yticks([])
plt.show()

# Automatic EDGE Detection
def auto_canny_edge_detection(im, sigma=20):
    md = np.median(im)
    lower_val = int(max(0, (1.0-sigma)*md))
    upper_val = int(min(255, (1.0+sigma)*md))
    return cv2.Canny(im, lower_val, upper_val)

auto_edge = auto_canny_edge_detection(blurred)

# Visualizing Auto Edged Image
plt.subplot(121),plt.imshow(blurred,cmap = 'gray')
plt.title('Original Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(auto_edge,cmap = 'gray')
plt.title('EDGED Image'), plt.xticks([]), plt.yticks([])
plt.show()






















