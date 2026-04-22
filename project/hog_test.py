# classification of potato leaf disease
"""Imports"""
import functions as f
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog

img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_early = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_late = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')


hog_features_healthy, hog_healthy = hog(
    img_healthy,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2',
    visualize=True,
    feature_vector=True
)

hog_features_early, hog_early = hog(
    img_early,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2',
    visualize=True,
    feature_vector=True
)

hog_features_late, hog_late = hog(
    img_late,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2',
    visualize=True,
    feature_vector=True
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

ax1.imshow(hog_healthy, cmap='gray')
ax1.set_title('HOG healthy')
ax1.axis('off')

ax2.imshow(hog_early, cmap='gray')
ax2.set_title('HOG early')
ax2.axis('off')

ax3.imshow(hog_late, cmap='gray')
ax3.set_title('HOG late')
ax3.axis('off')

plt.tight_layout()
plt.show()




