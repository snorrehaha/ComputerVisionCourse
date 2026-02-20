import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
###Always have to show the image as an int
###Do calculations on the image as a float
###You cant index when it is a float


def CalculateHistogram(img):
    M, N = img.shape[:2]
    histo = np.zeros(256, dtype=int)

    for i in range(M):
        for j in range(N):
            histo[int(img[i,j])] += 1
    return histo

def CDFCalc(histo):
    cdf = np.zeros(len(histo), dtype=int)
    i = 1
    cdf[0] = histo[0]
    for i in range(len(histo)):
        cdf[i] = cdf[i-1] + histo[i]
    return cdf

# Works only on uint8
def EqualiseImg(img):
    histo = CalculateHistogram(img)
    cdf = CDFCalc(histo)
    M, N = img.shape[:2]
    cdf_min = cdf[0]

    equalisation = np.round(((cdf - cdf_min) / ((M*N) - cdf_min)) * (256-1))
    eqImg = equalisation[img.astype(np.uint8)]
    return eqImg

# Loading image
alpha = 0.5
img = cv2.imread('../images/256x256_grayscale.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
img = img * alpha


# Create Histogram
histo = CalculateHistogram(img)
# Calculate cdf
cdf = CDFCalc(histo)
# Generate equalised image
eqImg = EqualiseImg(img)
# Equalised Histogram
eqHisto = CalculateHistogram(eqImg)


""" cdf histogram of image
plt.figure()
plt.plot(cdf, label='CDF')
plt.title('Histogram cdf')
plt.show()
"""
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- RAD 1: BILDER ---

# Originalbilde
axs[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axs[0, 0].set_title('Original (Alpha 0.5)')
axs[0, 0].axis('off') # Skjul akser for bilder

# Equalized bilde
axs[0, 1].imshow(eqImg, cmap='gray', vmin=0, vmax=255)
axs[0, 1].set_title('Etter Histogram Equalization')
axs[0, 1].axis('off')

# --- RAD 2: HISTOGRAMMER ---

x = np.arange(256)

# Originalt Histogram
axs[1, 0].bar(x, histo, color='blue', width=1.0)
axs[1, 0].set_title('Histogram (Original)')
axs[1, 0].set_xlim([0, 256])

# Nytt Histogram (nå strukket ut)
axs[1, 1].bar(x, eqHisto, color='blue', width=1.0)
axs[1, 1].set_title('Histogram (Equalized)')
axs[1, 1].set_xlim([0, 256])

# Juster avstand mellom plott og vis
plt.tight_layout()
plt.show()

#def visualizeHistogram(histogram):



cv2.destroyAllWindows()
