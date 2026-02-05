import numpy as np
import cv2
import matplotlib.pyplot as plt


rng = np.random.default_rng()

my_array = rng.integers(low=1, high=25, size=25)
matrix5x5 = np.matrix(my_array,int).reshape(5,5)
matrix3x3 = matrix5x5[np.ix_([1,2,3],[1,2,3])]
matrixTimes2 = matrix3x3 * 2
matrix5x5[1:4, 1:4] = matrixTimes2 # extract 3x3 in the middle
vector = matrix5x5.flatten()
r_matrix = np.matrix(np.flip(vector)).reshape(5,5)
print(r_matrix)


filePath = "original_image.jpg"
img = cv2.imread(filePath)

# area-based interpolation
imgArea = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
#cv2.imshow("Original image", imgArea)

#nearest neighbor interpolation
imgNearest = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_NEAREST)
#cv2.imshow("Original image", imgNearest)

#Grayscale
imgGray = cv2.cvtColor(imgArea, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Graysccale image", imgGray)

#rotated 180 degrees
imgRotated = cv2.rotate(imgArea, cv2.ROTATE_180)
#cv2.imshow("Rotated image", imgRotated)


#Converting and splitting channels
imgHSV = cv2.cvtColor(imgArea, cv2.COLOR_BGR2HSV)
imgXYZ = cv2.cvtColor(imgArea, cv2.COLOR_BGR2XYZ)
imgCIELAB = cv2.cvtColor(imgArea, cv2.COLOR_BGR2LAB)
H, S, V = cv2.split(imgHSV)
X, Y, Z = cv2.split(imgXYZ)
L, a, b = cv2.split(imgCIELAB)

"""
Hn = cv2.normalize(H, None, 0, 255, cv2.NORM_MINMAX)
Sn = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX)
Vn = cv2.normalize(V, None, 0, 255, cv2.NORM_MINMAX)

#cv2.imshow("Hue (normalized)", Hn)
#cv2.imshow("Saturation (normalized)", Sn)
#cv2.imshow("Value (normalized)", Vn)
"""

""" # XYZ colour space and histogram plot
Xn = cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX)
Yn = cv2.normalize(Y, None, 0, 255, cv2.NORM_MINMAX)
Zn = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("X (normalized)", Xn)
cv2.imshow("Y (normalized)", Yn)
cv2.imshow("Z (normalized)", Zn)

XHist = cv2.calcHist([X], [0], None, [256], [0, 256])
YHist = cv2.calcHist([Y], [0], None, [256], [0, 256])
ZHist = cv2.calcHist([Z], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(XHist)
plt.title("X channel histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")

plt.subplot(1, 3, 2)
plt.plot(YHist)
plt.title("Y channel histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")

plt.subplot(1, 3, 3)
plt.plot(ZHist)
plt.title("Z channel histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")

plt.tight_layout()
plt.show()
"""

"""
 #LAB colour space and histogram plot
Ln = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
An = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
Bn = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("An (normalized)", An)
cv2.imshow("Bn (normalized)", Bn)
cv2.imshow("L (normalized)", Ln)
LHist = cv2.calcHist([L], [0], None, [256], [0, 256])
aHist = cv2.calcHist([a], [0], None, [256], [0, 256])
bHist = cv2.calcHist([b], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(LHist)
plt.title("L channel histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")

plt.subplot(1, 3, 2)
plt.plot(aHist)
plt.title("a channel histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")

plt.subplot(1, 3, 3)
plt.plot(bHist)
plt.title("b channel histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel count")

plt.tight_layout()
plt.show()
"""

"""
#Excersize 5
# N = number of interpolation steps
N = 256

# 1
c1_rgb = np.array([255, 0, 0], dtype=np.float32)   # red
c2_rgb = np.array([0, 0, 255], dtype=np.float32)   # blue

#2
ramp_rgb = np.zeros((1, N, 3), dtype=np.float32)

for i, t in enumerate(np.linspace(0, 1, N)):
    ramp_rgb[0, i] = (1 - t) * c1_rgb + t * c2_rgb

#3
c1_lab = cv2.cvtColor(c1_rgb.reshape(1,1,3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
c2_lab = cv2.cvtColor(c2_rgb.reshape(1,1,3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

ramp_lab = np.zeros((1, N, 3), dtype=np.float32)

for i, t in enumerate(np.linspace(0, 1, N)):
    ramp_lab[0, i] = (1 - t) * c1_lab + t * c2_lab

ramp_lab_to_rgb = cv2.cvtColor(ramp_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

#results
plt.figure(figsize=(10,4))

for ch, color, label in zip(range(3), ['r','g','b'], ['R','G','B']):
    plt.plot(ramp_rgb[0,:,ch], color=color, linestyle='--', label=f'{label} (RGB interp)')
    plt.plot(ramp_lab_to_rgb[0,:,ch], color=color, label=f'{label} (LAB interp)')

plt.xlabel("Position")
plt.ylabel("Channel value")
plt.legend()
plt.title("RGB channel values along the ramp")
plt.show()
"""

warm_img = cv2.imread("../images/warm_picture.jpg").astype(np.float32)

cv2.imshow("Original", warm_img.astype(np.uint8))

# Compute mean of each channel (BGR)
mean_b, mean_g, mean_r, _ = cv2.mean(warm_img)

# Gray-world average
mean_gray = (mean_b + mean_g + mean_r) / 3

# Correct scaling factors
scale_b = mean_gray / mean_b
scale_g = mean_gray / mean_g
scale_r = mean_gray / mean_r

# Split channels
B, G, R = cv2.split(warm_img)

# Apply scaling
B *= scale_b
G *= scale_g
R *= scale_r

# Merge, clip, convert back
balanced_img = cv2.merge([B, G, R])
balanced_img = np.clip(balanced_img, 0, 255).astype(np.uint8)

cv2.imshow("Gray-World Balanced", balanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#Shape and type
print(img.shape)
print(imgGray.shape)
print(imgGray.dtype)
print(imgGray.min(), imgGray.max())


