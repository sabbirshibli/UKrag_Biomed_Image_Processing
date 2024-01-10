# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 00:07:58 2024

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

# Segmentation using K-Means Clustering
# Source: https://medium.com/towardssingularity/k-means-clustering-for-image-segmentation-using-opencv-in-python-17178ce3d6f3

# Reading Image
img = cv2.imread('image.jpg')
image = img[:, 50:420]  # Selecting SubImage on ROI

# Changing color to RGB from BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
pixel_vals = image.reshape((-1,3)) # numpy reshape operation -1 unspecified 

# Convert to float type only for supporting cv2.kmean
pixel_vals = np.float32(pixel_vals)

## Applying in-built Kmeans
'''
Mainly it has 5 arguments:
    1. samples: pixel values in np.float32
    2. nclusters(K): number of clusters
    3. criteria: iteration termination criteria. it has 3 params:
        a1) cv.TERM_CRITERIA_EPS = stop the algorithm iteration if specified accuracy (epsilon) is reached.
        a2) cv.TERM_CRITERIA_MAX_ITER = stop the algorithm after the specified number of iterations.
        a) cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER = stop the iteration when any of the above condition is met.
        b) max_iter = An integer specifying maximum number of iterations.
        c) epsilon = Required accuracy
    4. attempts: Flag to specify the number of times the algorithm is executed.
    5. flags: This flag is used to specify how initial centers are taken. Normally two flags are used for this:
        a) cv.KMEANS_PP_CENTERS
        b) cv.KMEANS_RANDOM_CENTERS.

cv2.kmeans returns following values:
    1. compactness: it is the sum of square distance from each point to their corresponding centers.
    2. labels: array of lavels (labels denote which pixel belongs to which cluster).
    3. centers: array of center values of clusters
'''

# Criteria construction
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# Defining number of Clusters (K)
K = 2

retval, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert data into 8-bit values
centers = np.uint8(centers)

# Mapping labels to center points (RGB)
segmented_data = centers[labels.flatten()]

# Reshape data into original image dimensions
segmented_image = segmented_data.reshape((image.shape))

# Visualizing Segmented Image
plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(segmented_image,cmap = 'gray')
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()












