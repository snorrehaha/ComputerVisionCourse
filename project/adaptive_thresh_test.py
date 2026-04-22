"""Imports"""
import functions as f
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import exposure

"""
Used for testing different thresholding 
Otsus seem like the most promising, but it still removes more of the leafs, than i would have liked
adaptive thresholding is basically useless
"""

img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_early = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_late = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')

images = [img_healthy, img_early, img_late]
masks = [f.segment_plant_hsv(img) for img in images]
segmented = [f.apply_mask(img, mask) for img, mask in zip(images, masks)]
titles = ["Healthy", "Early Blight", "Late Blight"]

f.show_segmentation_grid(images, masks, segmented, titles)
f.plot_hsv_histograms(segmented, titles)
f.plot_rgb_histograms(segmented, titles)

