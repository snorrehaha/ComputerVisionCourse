import functions as f
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_early   = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_late    = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')


def compute_elbow(img):
    h, w, c = img.shape
    data = img.reshape(-1, 3).astype(np.float32)

    k_range = range(1, 13)
    inertia = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    for k in k_range:
        compactness, labels, centers = cv2.kmeans(
            data,
            k,
            None,
            criteria,
            5,  # more attempts = more stable
            cv2.KMEANS_PP_CENTERS
        )
        inertia.append(compactness)

    return k_range, inertia


# Compute for each image
k_h, inertia_h = compute_elbow(img_healthy)
k_e, inertia_e = compute_elbow(img_early)
k_l, inertia_l = compute_elbow(img_late)


# Plot
plt.figure(figsize=(8, 5))

plt.plot(k_h, inertia_h, marker='o', label='Healthy')
plt.plot(k_e, inertia_e, marker='o', label='Early blight')
plt.plot(k_l, inertia_l, marker='o', label='Late blight')

plt.xlabel("Number of clusters K")
plt.ylabel("Inertia (compactness)")
plt.title("Elbow Plot for All Images")
plt.legend()
plt.grid(True)

plt.show()