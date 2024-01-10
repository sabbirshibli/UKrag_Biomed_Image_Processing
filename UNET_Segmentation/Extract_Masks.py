# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 03:02:28 2024

@author: Sabbir Ahmed Sibli
"""

import cv2
import numpy as np
import os

def extract_green_mask(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".jpg"):
            # Read the image
            image_path = os.path.join(input_folder, file_name)
            original_image = cv2.imread(image_path)
            # Convert the image to HSV color space
            hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
            
            # Define the range of green color in HSV
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])

            # Create a mask for the green color
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

            # Save the mask as a new image in the output folder
            mask_output_path = os.path.join(output_folder, f"mask_{file_name}")
            cv2.imwrite(mask_output_path, mask)

            # Save the masked image as a new image in the output folder
            masked_image_output_path = os.path.join(output_folder, f"masked_{file_name}")
            cv2.imwrite(masked_image_output_path, masked_image)

            print(f"Processed {file_name}")

# Provide the path to the input folder containing the JPG images
input_folder_path = "D:/Course Materials [Erasmus MSc]/University of Kragujevac/Biomedical Image Processing/Assignments/Seminar Paper/labeled_spines"

# Provide the path to the output folder where you want to save the masks and masked images
output_folder_path = "D:/Course Materials [Erasmus MSc]/University of Kragujevac/Biomedical Image Processing/Assignments/Seminar Paper/spine_masks"

# Call the function to extract green masks
extract_green_mask(input_folder_path, output_folder_path)
