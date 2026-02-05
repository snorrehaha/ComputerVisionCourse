import cv2
import numpy as np

# Load grayscale image
img = cv2.imread("warm_picture.jpg", cv2.IMREAD_GRAYSCALE)

# --- Exposure adjustment ---
alpha = 0.5
exposed = np.clip(img * alpha, 0, 255).astype(np.uint8)

# --- Histogram equalization ---
equalized = cv2.equalizeHist(exposed)

# --- Function to draw histogram as image ---
def draw_histogram(hist, width=512, height=400):
    hist_img = np.ones((height, width), dtype=np.uint8) * 255
    hist = hist.flatten()
    cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
    bin_width = int(width / 256)
    for i in range(1, 256):
        cv2.line(
            hist_img,
            (bin_width * (i-1), height - int(hist[i-1])),
            (bin_width * i, height - int(hist[i])),
            color=0,
            thickness=2
        )
    return hist_img

# --- Compute histograms ---
hist_exposed = cv2.calcHist([exposed], [0], None, [256], [0, 256])
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# --- Draw histogram images ---
hist_img_exposed = draw_histogram(hist_exposed)
hist_img_equalized = draw_histogram(hist_equalized)

# --- Show images and histograms ---
cv2.imshow("Exposed Image", exposed)
cv2.imshow("Exposed Histogram", hist_img_exposed)
cv2.imshow("Equalized Image", equalized)
cv2.imshow("Equalized Histogram", hist_img_equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()
