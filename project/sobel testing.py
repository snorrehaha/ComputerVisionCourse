import cv2
import numpy as np
import functions as f

def GetSobelFeatures(img):
    """
    Sobel-based edge and texture features.
    """

    L, a, b = cv2.split(img)
    gray = L

    # gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # orientation (use histogram instead of raw mean)
    orientation = np.arctan2(grad_y, grad_x)

    # normalize orientation to [0, π]
    orientation = np.abs(orientation)

    # features
    features = [
        np.mean(magnitude),
        np.std(magnitude),

        # better orientation representation
        np.std(orientation),
        np.mean(np.histogram(orientation, bins=8)[0])  # directional distribution
    ]

    return np.array(features)

import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowSobelOverlay(img_hsv, mask, title="Sobel Overlay"):
    """
    Visualize Sobel edges on top of original image.
    """

    # grayscale
    L, a, b = cv2.split(img_hsv)
    gray = L



    # Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # normalize for visualization
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_norm = mag_norm.astype(np.uint8)

    # convert original for overlay
    base = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    # heatmap
    heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # blend
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)

    # plot
    plt.figure(figsize=(12,5))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(base)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Sobel Magnitude")
    plt.imshow(mag_norm, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(title)
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_late = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_early = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')

images = [img_healthy, img_early, img_late]
titles = ["Healthy", "Early Blight", "Late Blight"]

for img, title in zip(images, titles):
    mask = f.segment_plant_hsv(img)
    ShowSobelOverlay(img, mask, title)

sobel_features = []

for img, title in zip(images, titles):



    feats = GetSobelFeatures(img)

    print(f"\n{title} Sobel Features:")
    for i, v in enumerate(feats):
        print(f"Feature {i}: {v:.4f}")

    sobel_features.append(feats)