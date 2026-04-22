import functions as f
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_early   = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_late    = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')


def kmeans_segment(img, k=4):
    h, w, c = img.shape

    data = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    centers = np.uint8(centers)

    segmented = centers[labels.flatten()].reshape((h, w, 3))
    labels_img = labels.reshape(h, w)

    return segmented, labels_img


# Segment images
seg_healthy, labels_h = kmeans_segment(img_healthy, 4)
seg_early, labels_e = kmeans_segment(img_early, 4)
seg_late, labels_l = kmeans_segment(img_late, 4)


# Convert originals to displayable format if needed
def to_display(img):
    if img.dtype != np.uint8:
        return (img * 255).astype(np.uint8)
    return img

img_healthy_disp = to_display(img_healthy)
img_early_disp   = to_display(img_early)
img_late_disp    = to_display(img_late)


# Plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Top row: originals
axes[0, 0].imshow(img_healthy_disp)
axes[0, 0].set_title("Healthy (Original)")
axes[0, 0].axis('off')

axes[0, 1].imshow(img_early_disp)
axes[0, 1].set_title("Early (Original)")
axes[0, 1].axis('off')

axes[0, 2].imshow(img_late_disp)
axes[0, 2].set_title("Late (Original)")
axes[0, 2].axis('off')


# Bottom row: segmented
axes[1, 0].imshow(cv2.cvtColor(seg_healthy, cv2.COLOR_LAB2RGB))
axes[1, 0].set_title("Healthy (K=4)")
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(seg_early, cv2.COLOR_LAB2RGB))
axes[1, 1].set_title("Early (K=4)")
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(seg_late, cv2.COLOR_LAB2RGB))
axes[1, 2].set_title("Late (K=4)")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()