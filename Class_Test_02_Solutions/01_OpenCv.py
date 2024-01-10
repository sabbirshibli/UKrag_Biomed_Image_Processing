# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:01:43 2024

@author: Sabbir Ahmed Sibli
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
image = cv2.imread('test_image.png')

# Rotating the image
M = cv2.getRotationMatrix2D((512/2,512/2), 45, 1)
rotated = cv2.warpAffine(image, M, (512,512))

plt.imshow(rotated)
plt.show()

# Cropping the image
cropped = image[50:280, 150:350]

plt.imshow(cropped)
plt.plot()

# Resize the image (half of the original shape)
resized = cv2.resize(image, fx=.5,fy=.5,dsize=None)

plt.imshow(resized)
plt.show()

# Convert image to grayscale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(img_gray, cmap='gray')
plt.show()

# Plotting the histogram
plt.subplot(121)
plt.title("Original Image")
plt.imshow(img_gray, cmap='gray')
plt.subplot(122)
plt.title("Histogram")
plt.hist(img_gray.ravel(),256,[0,255])
plt.plot()

# Adjust Image Contast
img_contrast = cv2.equalizeHist(img_gray)

plt.subplot(121)
plt.title('Original Image')
plt.imshow(img_gray, cmap='gray')
plt.subplot(122)
plt.title("Adjusted Contrast Image")
plt.imshow(img_contrast, cmap='gray')
plt.plot()

# Blur image using Gaussian Blur

img_blurry = cv2.GaussianBlur(img_gray, (5,5), 0)

plt.subplot(121)
plt.title("Original Image")
plt.imshow(img_gray, cmap='gray')
plt.subplot(122)
plt.title("Blurry Image")
plt.imshow(img_blurry, cmap='gray')
plt.plot()

# Canny Edge Detection
from skimage.feature import canny
img_canny = canny(img_blurry)

plt.imshow(img_canny, cmap='gray')
plt.plot()

# Erosion and Dilation (The output is not working)
kernel = np.ones((5,5), np.uint8)
# eroted = cv2.erode(img_canny, kernel, iterations=5)     # This is another way to apply erode
eroted =  cv2.morphologyEx(img_canny, cv2.MORPH_ERODE, kernel, iterations=5)
plt.imshow(eroted, cmap='gray')
plt.plot()

dilated = cv2.morphologyEx(eroted, cv2.MORPH_DILATE, kernel, iterations=5)
plt.imshow(dilated, cmap='gray')
plt.plot()

# Otsu Segmentation
(T, thresh_otsu) = cv2.threshold(img_canny,100,256,cv2.THRESH_OTSU|cv2.THRESH_BINARY)

plt.imshow(thresh_otsu)
plt.plot()










