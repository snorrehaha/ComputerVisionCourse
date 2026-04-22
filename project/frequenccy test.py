import cv2
import numpy as np
import matplotlib.pyplot as plt
import functions as f

def fft_features_and_visualize(img_hsv, mask, title="Image"):
    """
    1. Apply mask (focus only on leaf)
    2. Compute FFT
    3. Visualize magnitude spectrum
    4. Extract simple + band features
    """

    # --- grayscale from HSV (L channel proxy) ---
    gray = img_hsv[:, :, 2]

    # --- apply segmentation mask ---
    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0

    # --- FFT ---
    f = np.fft.fft2(gray_masked)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log(1 + magnitude)

    # --- feature extraction (your original + extended) ---
    mean_val = np.mean(log_mag)
    std_val = np.std(log_mag)

    # energy ratio (low vs high freq approximation)
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    radius = min(cy, cx) // 2

    low_freq = log_mag[cy-radius:cy+radius, cx-radius:cx+radius]
    low_energy = np.mean(low_freq)

    high_energy = np.mean(log_mag) - low_energy

    features = np.array([mean_val, std_val, low_energy, high_energy])

    # --- visualization ---
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title(f"{title} - Masked Leaf")
    plt.imshow(gray_masked, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title(f"{title} - FFT Spectrum")
    plt.imshow(log_mag, cmap='inferno')
    plt.axis('off')

    plt.show()

    print(f"\n{title} Frequency Features:")
    print(f"Mean spectrum:      {mean_val:.4f}")
    print(f"Std spectrum:       {std_val:.4f}")
    print(f"Low freq energy:    {low_energy:.4f}")
    print(f"High freq energy:   {high_energy:.4f}")

    return features

img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_late = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_early = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')


images = [img_healthy, img_early, img_late]
titles = ["Healthy", "Early Blight", "Late Blight"]

for img, title in zip(images, titles):
    mask = f.segment_plant_hsv(img)
    feats = fft_features_and_visualize(img, mask, title)