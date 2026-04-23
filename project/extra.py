import kagglehub

# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)

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


"""FUNCTIONS for experimenting, that was not needed for the final version of the model"""

"""Split HSV image into channels"""
def split_hsv(img):
    return cv2.split(img)

"""Binary threshold on S channel"""
def threshold_saturation(s_channel, thresh=50):
    _, mask = cv2.threshold(s_channel, thresh, 255, cv2.THRESH_BINARY)
    return mask

"""Apply closing then opening to clean noise"""
def morph_cleanup(mask, kernel, close_iter=3, open_iter=2):
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return opened

"""Keep only the largest connected component"""
def largest_contour_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask, dtype=np.uint8)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return clean_mask.astype(np.uint8)

"""
Remove near-black shadow pixels from the leaf mask.
Pixels with V < v_thresh are considered shadow and are excluded.
"""
def exclude_dark_shadows(mask, v_channel, v_thresh=40):
    bright_enough = (v_channel > v_thresh).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, bright_enough)

"""
Full pipeline:
HSV -> S-threshold -> Shadow exclusion -> morphology -> largest contour
"""
def segment_plant_hsv(img_hsv, sat_threshold=50, v_shadow_thresh=40, kernel_size=5, close_iter=3, open_iter=2):
    _, s, v = split_hsv(img_hsv)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = threshold_saturation(s, sat_threshold)
    mask = exclude_dark_shadows(mask, v, v_shadow_thresh)
    mask = morph_cleanup(mask, kernel, close_iter, open_iter)
    #mask = largest_contour_mask(mask)

    return mask.astype(np.uint8)

"""WATERSHED experiments"""
"""Remove noise from the mask"""
def build_disease_mask(img_hsv, leaf_mask, kernel_size=3):
    early, late_yellow, late_brown = disease_submasks(img_hsv, leaf_mask)

    combined = cv2.bitwise_or(early, late_yellow)
    combined = cv2.bitwise_or(combined, late_brown)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Slightly conservative cleanup (preserve small spots)
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    return cleaned, {"early": early, "late_yellow": late_yellow, "late_brown": late_brown}


"""Create markers for watershed using distance transform"""
def watershed_spots(img_hsv, clean_mask, dist_thresh=0.4):
    """
    Apply watershed to split individual spots.
    """

    kernel = np.ones((3,3), np.uint8)

    # Background
    sure_bg = cv2.dilate(clean_mask, kernel, iterations=3)

    # Distance transform
    dist = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)

    # Safety check (important)
    if dist.max() == 0:
        empty = np.zeros_like(clean_mask, dtype=np.int32)
        return empty, sure_bg, None, None

    # Foreground
    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    markers = cv2.watershed(img_bgr, markers)

    return markers, sure_bg, sure_fg, unknown

"""Apply watershed using BGR image"""
def filter_small_spots(markers, min_area=50):
    """
    Remove tiny regions caused by over-segmentation.
    """

    labeled = np.where(markers > 1, markers, 0).astype(np.int32)

    for lbl in np.unique(labeled):
        if lbl == 0:
            continue

        if np.sum(labeled == lbl) < min_area:
            labeled[labeled == lbl] = 0

    clean_mask = (labeled > 0).astype(np.uint8) * 255
    count = len(np.unique(labeled)) - 1

    return labeled, clean_mask, count

"""
Full pipeline:
Leaf mask -> disease mask -> cleanup -> watershed -> result
"""
def detect_blight_watershed(img_hsv, leaf_mask,
                           dist_thresh=0.4,
                           min_area=50):
    """
    Full pipeline:
    HSV → submasks → clean → watershed → filter
    """

    # Step 1: disease mask
    clean_mask, submasks = build_disease_mask(img_hsv, leaf_mask)

    # Step 2: watershed
    markers, sure_bg, sure_fg, unknown = watershed_spots(
        img_hsv, clean_mask, dist_thresh
    )

    # Step 3: filter small regions
    labeled, spot_mask, count = filter_small_spots(markers, min_area)

    # Step 4: visualization
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    result = img_bgr.copy()
    result[markers == -1] = [0, 0, 255]

    return {
        "result": result,
        "count": count,
        "spot_mask": spot_mask,
        "markers": labeled,
        "clean_mask": clean_mask,
        "submasks": submasks,
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "unknown": unknown
    }

"""DISPLAY-----------------------------------------------------------------------------------------------------------"""
"""Apply binary mask to image"""
def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))

"""Convert HSV image to RGB for display"""
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

"""
Displays:
Row 1: original images
Row 2: binary masks
Row 3: segmented images (largest contour applied)
"""
def show_segmentation_grid(images, masks, seg_masks, titles):
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))

    for i in range(n):

        # --- Original ---
        axes[0, i].imshow(to_rgb(images[i]))
        axes[0, i].set_title(f"{titles[i]} - Original")
        axes[0, i].axis("off")

        # --- Mask ---
        axes[1, i].imshow(masks[i], cmap="gray")
        axes[1, i].set_title(f"{titles[i]} - Mask")
        axes[1, i].axis("off")

        # --- Segmented ---
        rgb = to_rgb(images[i])
        mask = seg_masks[i]
        # --- enforce correct dtype ---
        mask = mask.astype(np.uint8)
        # --- enforce single channel ---
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        # --- FORCE size match  --- While it should work without, there seem to be small errors without the resizing, making it unable to compile
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
        segmented = cv2.bitwise_and(rgb, rgb, mask=mask)

        axes[2, i].imshow(segmented)
        axes[2, i].set_title(f"{titles[i]} - Segmented")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()

"""
Plots H, S, V histograms for segmented images
"""
def plot_hsv_histograms(segmented_images, titles):
    fig, axes = plt.subplots(3, len(segmented_images), figsize=(15, 10))

    for i, img in enumerate(segmented_images):

        # Ensure valid image
        hsv = img.copy()

        # Split channels
        h, s, v = cv2.split(hsv)

        #Excludes the 0 value, from the segmentation
        # --- H channel ---
        axes[0, i].hist(h.ravel(), bins=50, color='orange', range=(1,h.max()))
        axes[0, i].set_title(f"{titles[i]} - Hue")
        axes[0, i].set_xlim([0, 180])

        # --- S channel ---
        axes[1, i].hist(s.ravel(), bins=50, color='green', range=(1, s.max()))
        axes[1, i].set_title(f"{titles[i]} - Saturation")
        axes[1, i].set_xlim([0, 255])

        # --- V channel ---
        axes[2, i].hist(v.ravel(), bins=50, color='gray', range=(1, v.max()))
        axes[2, i].set_title(f"{titles[i]} - Value")
        axes[2, i].set_xlim([0, 255])

    plt.tight_layout()
    plt.show()


"""
Plots R, G, B histograms for segmented images
"""
def plot_rgb_histograms(segmented_images, titles):

    fig, axes = plt.subplots(3, len(segmented_images), figsize=(15, 10))

    for i, img in enumerate(segmented_images):

        # Convert HSV → RGB (important: your pipeline uses HSV)
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        r, g, b = cv2.split(rgb)

        # --- R channel ---
        axes[0, i].hist(r.ravel(), bins=50, color='red')
        axes[0, i].set_title(f"{titles[i]} - Red")
        axes[0, i].set_xlim([0, 255])

        # --- G channel ---
        axes[1, i].hist(g.ravel(), bins=50, color='green')
        axes[1, i].set_title(f"{titles[i]} - Green")
        axes[1, i].set_xlim([0, 255])

        # --- B channel ---
        axes[2, i].hist(b.ravel(), bins=50, color='blue')
        axes[2, i].set_title(f"{titles[i]} - Blue")
        axes[2, i].set_xlim([0, 255])

    plt.tight_layout()
    plt.show()

def plot_images_grid(images, titles, cols=4, figsize=(12,6)):
    rows = int(np.ceil(len(images) / cols))
    plt.figure(figsize=figsize)

    for i, (img, title) in enumerate(zip(images, titles)):
        #img = to_rgb(img)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()