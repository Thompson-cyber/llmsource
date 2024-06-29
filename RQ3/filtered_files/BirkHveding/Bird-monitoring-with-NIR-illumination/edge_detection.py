# This script was writen by ChatGPT and modified by the author of this repo.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def otsu_and_detect_edges(image):
    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply Canny edge detection
    edges = cv2.Canny(thresholded, 100, 200)

    return thresholded, edges

def visualize_images(original, thresholded, edges, title):
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the original image
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original')

    # Plot the Otsu thresholded image
    axs[1].imshow(thresholded, cmap='gray')
    axs[1].set_title('Otsu Thresholded')

    # Plot the edge image
    axs[2].imshow(edges, cmap='gray')
    axs[2].set_title('Canny Edges')

    # Remove ticks and labels from subplots
    for ax in axs:
        ax.axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # set figure name equal to image name
    fig.suptitle(title)

    # Display the figure
    plt.show()

def equalize_and_detect_edges(image):
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(image)

    # Apply Canny edge detection
    edges = cv2.Canny(equalized_image, 100, 200)

    return equalized_image, edges

def iterate_through_folder(path):
    # functon for iterating through all files in a folder   
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file_path)

        # Load the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Apply Otsu's thresholding and edge detection
        thresholded_image, edges = otsu_and_detect_edges(image)

        # Visualize the original, Otsu thresholded, and edge images
        visualize_images(image, thresholded_image, edges)

file_path = r'path'