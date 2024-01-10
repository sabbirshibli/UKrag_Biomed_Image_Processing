# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 23:17:31 2024

@author: Sabbir Ahmed Sibli
"""
# Importing Important Libraries
from pathlib import Path
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dicom2jpg
from PIL import Image

# Segmentation using Thresholding Methods
# Source: https://regenerativetoday.com/how-to-perform-image-segmentation-with-thresholding-using-opencv/

# Reading Image
img = cv2.imread('image.jpg')
image = img[:, 50:420]  # Selecting SubImage on ROI

# Applying Gaussian Smoothing/Blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7,7), 0)

## Applying Simple Thresholding (Binary Thresholding)
# cv2.threshold(image, threshold_value, upper_pixel_value, threshold_method)
(T, thresh) = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

# Visualizing the Segmented Image
plt.subplot(121),plt.imshow(blurred,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(thresh,cmap = 'gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()

## Otsu Thresholding: Let OpenCV finds an optimal threshold value
(T, threshOtsu) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Printing the Otsu threshold value
print("Otsu threshold value:", T)  # in this case it's 47.0

# Visualizing the Segmented Image
plt.subplot(121),plt.imshow(blurred,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(threshOtsu,cmap = 'gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()

## Adaptive Thresholding
# cv2.adaptiveThreshold(image, upper_pixel_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, region_value(given N will work as NxN), random_constant)
threshAda = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255,-11.5)

# Visualizing the Segmented Image
plt.subplot(121),plt.imshow(blurred,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(threshAda,cmap = 'gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()
















