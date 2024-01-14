# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:42:59 2024

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

# Reading a single dicom file
dicom_file = pydicom.read_file('D:/Course Materials [Erasmus MSc]/University of Kragujevac/Biomedical Image Processing/Assignments/Datasets/Spine_DICOM/5DB02D1B')
# print(dicom_file)
# print(dicom_file.Rows)

actual_image = dicom_file.pixel_array # Actual image from the all dicom attributes

# Plotting the actual image
plt.figure()
plt.imshow(actual_image, cmap='gray')

# Reading all the dicom slices using pathlib library
path_to_spine = Path('D:/Course Materials [Erasmus MSc]/University of Kragujevac/Biomedical Image Processing/Assignments/Seminar Paper/Patient 1 - Spine/DICOM/')
all_files = list(path_to_spine.glob("*"))
print(all_files)

# Appending all slices into a list
ct_spine = []
for path in all_files:
    data = pydicom.read_file(path)
    # Skipping slices who doesn't have the Pixel Data / pixel_array attribute
    if "PixelData" in data:
        # print(f"Dataset found with Pixel Data: {path}")
        ct_spine.append(data)
    else:
        continue

'''
# skipping files with no SliceLocation (Sorting Purpose)
final_slices = []
skipcount = 0
for f in ct_spine:
    if hasattr(path, 'SliceLocation'):
        final_slices.append(f)
    else:
        skipcount += 1
'''

# Sorting the slices

# Checking wheather the slices are sorted or not
for slice in ct_spine[:10]:
    print(slice.SliceLocation)

# Ordering the slices (No need if sorted already)
# ct_spine_ordered = sorted(ct_spine, key=lambda slice: slice.SliceLocation)

# Creating Complete 3D image
full_volume = []
for slice in ct_spine:
    full_volume.append(slice.pixel_array)

# print(type(full_volume))

# Plotting some slices
fig, axis = plt.subplots(3,3, figsize=(10,10))
slice_counter = 0
for i in range(3):
    for j in range(3):
        axis[i][j].imshow(full_volume[slice_counter], cmap='gray')
        slice_counter += 1

# Plotting a single slice from the list
plt.figure()
plt.imshow(full_volume[512], cmap='gray')

# Converting a single Dicom to JPG
scaled_image = (np.maximum(full_volume[512], 0) / full_volume[512].max()) * 255.0
scaled_image = np.uint8(scaled_image)
final_image = Image.fromarray(scaled_image)
final_image.save('image.jpg')
