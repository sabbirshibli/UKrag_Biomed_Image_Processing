# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:16:42 2024

@author: Sabbir Ahmed Sibli
"""
# Source: https://medium.com/@sasasulakshi/opencv-image-histogram-calculations-4c5e736f85e

#%matplotlib inline
# Importing Important Libraries
from pathlib import Path
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dicom2jpg
from PIL import Image

# Reading the Image
img = cv2.imread('image.jpg')

image = img[:, 50:420]  # Selecting SubImage on ROI

# Visualizing Original Image
cv2.namedWindow("BGR Image", cv2.WINDOW_NORMAL)
cv2.imshow("BGR Image", image)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

# Separating Color Channels (Though my sample image is Grayscale)
B = image[:,:,0]  # Blue Layer
G = image[:,:,1]  # Green Layer
R = image[:,:,2]  # Red Layer

## Calculating Hists
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# image = Original image [image]
# channels = Color Channels (eg. BGR as [0], [1], [2])
# mask = mask is defined for applying histogram in a specific region. For full image it is "None"t
# histSize = BIN count
# ranges = range of pixel values. For 256x256 range will be [0,256]

B_hist = cv2.calcHist([image], [0], None, [256], [0,256])
G_hist = cv2.calcHist([image], [1], None, [256], [0,256])
R_hist = cv2.calcHist([image], [2], None, [256], [0,256])

# Visualizing Histograms
plt.subplot(2,2,1)
plt.plot(B_hist, 'b')
plt.subplot(2,2,2)
plt.plot(G_hist, 'g')
plt.subplot(2,2,3)
plt.plot(R_hist, 'r')

## Histogram Equalization: Distributing pixels to develop overexposed or underexposed image
# cv2.equalizeHist()

b_eq = cv2.equalizeHist(B)
g_eq = cv2.equalizeHist(G)
r_eq = cv2.equalizeHist(R)

# Visualizing equalized images
# For our case, each image will be the same as it is grayscale image
plt.imshow(b_eq)
plt.title("b_eq")
plt.show()
plt.imshow(g_eq)
plt.title("g_eq")
plt.show()
plt.imshow(r_eq)
plt.title("r_eq")
plt.show()

# Calculating Histogram for each colored image
B_hist = cv2.calcHist([b_eq],[0], None, [256], [0,256]) 
G_hist = cv2.calcHist([g_eq],[0], None, [256], [0,256])
R_hist = cv2.calcHist([r_eq],[0], None, [256], [0,256])

# Plotting the Histograms for each image
plt.subplot(2, 2, 1)
plt.plot(G_hist, 'g')
plt.subplot(2, 2, 2)
plt.plot(R_hist, 'r')
plt.subplot(2, 2, 3)
plt.plot(B_hist, 'b')

# Merging all equalized layers into original image
eq_im = cv2.merge([b_eq, g_eq, r_eq])

# Visualize equalized original image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL);
cv2.imshow("Original Image",image);
cv2.namedWindow("New Image", cv2.WINDOW_NORMAL);
cv2.imshow("New Image",eq_im);

cv2.waitKey(0) & 0xFF 
cv2.destroyAllWindows()























