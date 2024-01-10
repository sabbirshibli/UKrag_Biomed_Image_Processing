# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:51:48 2024

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

# Segmentation using Watershed Algorithm and OpenCV
# Source: https://medium.com/@jaskaranbhatia/exploring-image-segmentation-techniques-watershed-algorithm-using-opencv-9f73d2bc7c5a

# Load image
img = cv2.imread('image.jpg')
plt.axis('off')
plt.imshow(img)

# Thresholding
# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold using OTSU
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.axis('off')
plt.imshow(thresh)

# noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

plt.imshow(sure_fg)
plt.imshow(sure_bg)
plt.imshow(unknown)

# Marker labelling
# Connected Components determines the connectivity of blob-like regions in a binary image.
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.imshow(img)
