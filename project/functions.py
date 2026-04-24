import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import cv2
import numpy as np

"""MAIN SEGMENTATION ------------------------------------------------------------------------------------------------"""
"""Load image, resize, and convert to HSV."""
def LoadImage(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img





import numpy as np
import cv2

"""Segment plant using K-means clustering in HSV space"""


"""WATERSHED---------------------------------------------------------------------------------------------------------"""

EARLY_BLIGHT_LOWER       = np.array([ 9,  20,  26], dtype=np.uint8)
EARLY_BLIGHT_UPPER       = np.array([19, 195, 189], dtype=np.uint8)

# Late Blight yellow (#c9c561, #ced16a, #b9bb56)
#   Sampled HSV: H 29-31, S 126-138, V 187-209
LATE_BLIGHT_YELLOW_LOWER = np.array([25,  96, 157], dtype=np.uint8)
LATE_BLIGHT_YELLOW_UPPER = np.array([35, 168, 239], dtype=np.uint8)

# Late Blight brown (#23120a, #876450, #9e8d73)
#   Sampled HSV: H 10-18, S 69-182, V 35-158
LATE_BLIGHT_BROWN_LOWER  = np.array([ 6,  39,  15], dtype=np.uint8)
LATE_BLIGHT_BROWN_UPPER  = np.array([22, 212, 188], dtype=np.uint8)

# Fraction of spot pixels that must be yellow to call Late Blight
LATE_BLIGHT_YELLOW_RATIO_THRESH = 0.10

# Overlay colours per disease class (RGB)
SPOT_COLOURS = {
    "Healthy":      (  0, 200,  50),
    "Early Blight": (255, 200,   0),
    "Late Blight":  (180,  40,   0),
}

"""Keep only leaf pixels in image"""
def disease_submasks(img_hsv, leaf_mask):
    """
    Create masks using calibrated HSV ranges.
    """
    early = cv2.inRange(img_hsv,EARLY_BLIGHT_LOWER,EARLY_BLIGHT_UPPER)

    late_yellow = cv2.inRange(img_hsv,LATE_BLIGHT_YELLOW_LOWER,LATE_BLIGHT_YELLOW_UPPER)

    late_brown = cv2.inRange(img_hsv,LATE_BLIGHT_BROWN_LOWER,LATE_BLIGHT_BROWN_UPPER)

    # Apply leaf mask (critical)
    for m in (early, late_yellow, late_brown):
        cv2.bitwise_and(m, m, dst=m, mask=leaf_mask)

    return early, late_yellow, late_brown




"""FEATURE EXTRACTION COLOUR ----------------------------------------------------------------------------------------"""

"""Extract color-based features from leaf pixels"""
def extract_color_features(img_hsv, leaf_mask):
    h, s, v = cv2.split(img_hsv)

    # Only leaf pixels
    mask = leaf_mask > 0

    h_vals = h[mask]
    s_vals = s[mask]
    v_vals = v[mask]

    features = []

    # --- Basic statistics ---
    features.extend([
        np.mean(h_vals), np.std(h_vals),
        np.mean(s_vals), np.std(s_vals),
        np.mean(v_vals), np.std(v_vals),
    ])

    # --- Disease-related pixel ratios ---
    early, late_yellow, late_brown = disease_submasks(img_hsv, leaf_mask)

    total_pixels = np.sum(mask)

    early_ratio = np.sum(early > 0) / total_pixels
    yellow_ratio = np.sum(late_yellow > 0) / total_pixels
    brown_ratio = np.sum(late_brown > 0) / total_pixels

    features.extend([early_ratio, yellow_ratio, brown_ratio])

    return np.array(features, dtype=np.float32)

from skimage.feature import graycomatrix, graycoprops

"""Extract GLCM texture features from leaf region"""
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2

def extract_glcm_features(img_hsv, leaf_mask):

    # Convert HSV → grayscale
    bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    mask = leaf_mask > 0

    # Apply mask
    gray_leaf = gray.copy()
    gray_leaf[~mask] = 0

    # Quantise intensity
    gray_q = (gray_leaf / 32).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(
        gray_q,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32,
        symmetric=True,
        normed=True
    )

    # Core features
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    # Additional features
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    asm = graycoprops(glcm, 'ASM').mean()

    # Entropy (manual)
    glcm_eps = glcm + 1e-10
    entropy = -np.sum(glcm_eps * np.log(glcm_eps))

    return np.array([
        contrast, homogeneity, energy, correlation,
        dissimilarity, asm, entropy
    ], dtype=np.float32)

def extract_features(img_hsv, leaf_mask):

    color_feats = extract_color_features(img_hsv, leaf_mask)
    texture_feats = extract_glcm_features(img_hsv, leaf_mask)

    return np.concatenate([color_feats, texture_feats  ]) #color_feats, texture_feats

"""DISPLAY ----------------------------------------------------------------------------------------------------------"""
