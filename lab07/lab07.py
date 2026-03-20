#Lab 07
import numpy as np
import matplotlib.pyplot as plt
import cv2

# FUNCTIONS ----------------------------------------------------------------------------------------
def LoadImage(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.astype(np.float32) / 255.0
    return img

def GetSobelFeatures(img):
    # Use L channel for intensity/edges
    L, a, b = cv2.split(img)

    # Compute x and y gradients using Sobel
    Gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude and orientation
    M = np.sqrt(Gx ** 2 + Gy ** 2)
    Theta = np.arctan2(Gy, Gx)  # in radians

    return Gx, Gy, M, Theta


#Exercise 1 PART A -----------------------------------------------------------------------------------------

img = LoadImage('original_image.jpg')
Gx, Gy, M, Theta = GetSobelFeatures(img)
threshold = [30,80,150]

# Normalize for display
Gx = cv2.normalize(Gx, None, 0, 255, cv2.NORM_MINMAX)
Gy = cv2.normalize(Gy, None, 0, 255, cv2.NORM_MINMAX)
M = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX)
Theta = cv2.normalize(Theta, None, 0, 255, cv2.NORM_MINMAX)

Gx = Gx.astype(np.uint8)
Gy = (Gy.astype(np.uint8))
M = M.astype(np.uint8)
Theta = Theta.astype(np.uint8)

binary_maps = []
for i,t in enumerate(threshold):
    _, binary = cv2.threshold(M, t, 255, cv2.THRESH_BINARY)
    binary_maps.append(binary)

# Display
plt.figure(figsize=(12,8))

plt.subplot(2,4,1)
plt.imshow(Gx, cmap='gray')
plt.title("Gx")
plt.axis("off")

plt.subplot(2,4,2)
plt.imshow(Gy, cmap='gray')
plt.title("Gy")
plt.axis("off")

plt.subplot(2,4,3)
plt.imshow(M, cmap='gray')
plt.title("Magnitude")
plt.axis("off")

plt.subplot(2,4,4)
plt.imshow(Theta, cmap='gray')
plt.title("Theta")
plt.axis("off")

for i, binary in enumerate(binary_maps):
    plt.subplot(2,4,5+i)
    plt.imshow(binary, cmap='gray')
    plt.title(f"Threshold {threshold[i]}")
    plt.axis("off")

plt.show()








