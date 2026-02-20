import numpy as np
import cv2
import matplotlib.pyplot as plt

#load
img = cv2.imread("../images/warm_picture.jpg", cv2.IMREAD_COLOR_BGR).astype(np.float32) / 255.0
img = np.clip(img, 0, 1)

#calc
alpha = [0.25, 0.5, 1.0, 2.0, 4.0]

def generateImage(img):
    img_arr = []
    for a in alpha:
        I = np.clip(img * a, 0, 1)
        img_arr.append(I)
    return img_arr


def getLuminance(img):
    B, G, R = cv2.split(img)
    luminance = np.float32(0.299 * R + 0.587 * G + 0.114 * B)
    return luminance

imgArr = generateImage(img)

imgo = getLuminance(img)
img1 = getLuminance(imgArr[0])
img2 = getLuminance(imgArr[1])
img3 = getLuminance(imgArr[2])
img4 = getLuminance(imgArr[3])
img5 = getLuminance(imgArr[4])


# Plot
fig, axs = plt.subplots(6, 2, figsize=(12, 10))

#Original
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original")
axs[0,0].axis("off")

axs[0, 1].hist(imgo.ravel() * 255.0, 256, range=[0, 256])
axs[0, 1].set_title("Histogram")

#Exposure 0.25
axs[1, 0].imshow(cv2.cvtColor(imgArr[0], cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Exposure 0.25")
axs[1, 0].axis("off")

axs[1, 1].hist(img1.ravel() * 255.0, 256, range=[0, 256])
axs[1, 1].set_title("Histogram 0.25")

#Exposure 0.5
axs[2, 0].imshow(cv2.cvtColor(imgArr[1], cv2.COLOR_BGR2RGB))
axs[2, 0].set_title("Exposure 0.5")
axs[2, 0].axis("off")

axs[2, 1].hist(img2.ravel() * 255.0, 256, range=[0, 256])
axs[2, 1].set_title("Histogram 0.5")

#Exposure 1
axs[3, 0].imshow(cv2.cvtColor(imgArr[2], cv2.COLOR_BGR2RGB))
axs[3, 0].set_title("Exposure 1.0")
axs[3, 0].axis("off")

axs[3, 1].hist(img3.ravel() * 255.0, 256, range=[0, 256])
axs[3, 1].set_title("Histogram 1.0")

#Exposure 2
axs[4, 0].imshow(cv2.cvtColor(imgArr[3], cv2.COLOR_BGR2RGB))
axs[4, 0].set_title("Exposure 2.0")
axs[4, 0].axis("off")

axs[4, 1].hist(img4.ravel() * 255.0, 256, range=[0, 256])
axs[4, 1].set_title("Histogram 2.0")

#Exposure 4
axs[5, 0].imshow(cv2.cvtColor(imgArr[4], cv2.COLOR_BGR2RGB))
axs[5, 0].set_title("Exposure 4.0")
axs[5, 0].axis("off")

axs[5, 1].hist(img5.ravel() * 255.0, 256, range=[0, 256])
axs[5, 1].set_title("Histogram 4.0")

plt.tight_layout()
plt.show()

def crushed_count(img):
    return np.sum(img == 0)

def saturated_count(img):
    return np.sum(img == 1)

for i, I in enumerate(imgArr):
    print(i, crushed_count(I), saturated_count(I))

print(img.size)
