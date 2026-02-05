
"""import cv2
import numpy as np

img = cv2.imread("slothpng")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape
mask = np.zeros((h, w), dtype=np.uint8)

for i in range(h):
    for j in range(w):
        if gray[i, j] > 120:
            mask[i, j] = 255
        else:
            mask[i, j] = 0

cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

import cv2

# Read image
img = cv2.imread("slothpng")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Show result
cv2.imshow("Binary Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
