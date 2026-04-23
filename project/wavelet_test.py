import cv2
import numpy as np
import pywt
import functions as f

def GetWaveletFeatures(img_hsv):

    gray = img_hsv[:, :, 2]

    # wavelet decomposition
    coeffs = pywt.wavedec2(gray, wavelet='db1', level=2)

    features = []

    cA = coeffs[0]
    features.append(np.mean(cA))
    features.append(np.std(cA))

    for level in coeffs[1:]:
        for band in level:
            features.append(np.mean(band))
            features.append(np.std(band))

    return np.array(features)


img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_late = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_early = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')

images = [img_healthy, img_early, img_late]
titles = ["Healthy", "Early Blight", "Late Blight"]

wavelet_features = []

for img, title in zip(images, titles):

    # Step 1: segmentation (important)
    mask = f.segment_plant_hsv(img)
    segmented = f.apply_mask(img, mask)

    # Step 2: wavelet features
    feats = GetWaveletFeatures(segmented)

    # Step 3: print results
    print(f"\n{title} Wavelet Features:")
    for i, v in enumerate(feats):
        print(f"Feature {i}: {v:.4f}")

    wavelet_features.append(feats)