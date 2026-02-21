import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn
import os
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pywt

# Paths
bear_paths = [os.path.join("../Datasets/train/Bear", f) for f in os.listdir("../Datasets/train/Bear")]
crocodile_paths = [os.path.join("../Datasets/train/Crocodile", f) for f in os.listdir("../Datasets/train/Crocodile")]

# Load and preprocess image
def LoadImage(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.astype(np.float32) / 255.0
    return img

# Extract HOG features
def GetHog(img):
    L, a, b = cv2.split(img)
    hog_features = hog(L, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return hog_features

# Extract Fourier features (simple: mean & std of magnitude spectrum)
def GetFourierFeatures(img):
    # Convert LAB to grayscale approximation
    L, a, b = cv2.split(img)
    gray = L
    # Compute 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    # Take log to compress range
    magnitude = np.log1p(magnitude)
    # Simple statistics as features
    mean_val = np.mean(magnitude)
    std_val = np.std(magnitude)
    max_val = np.max(magnitude)
    min_val = np.min(magnitude)
    return np.array([mean_val, std_val, max_val, min_val])



# Combine features
X = []
Y = []

for path in bear_paths:
    img = LoadImage(path)
    hog_feat = GetHog(img)
    wavelenghts_feat = GetWaveletFeatures(img)
    combined_feat = np.concatenate([hog_feat, wavelenghts_feat])
    X.append(combined_feat)
    Y.append(0)

for path in crocodile_paths:
    img = LoadImage(path)
    hog_feat = GetHog(img)
    wavelenghts_feat = GetWaveletFeatures(img)
    combined_feat = np.concatenate([hog_feat, wavelenghts_feat])
    X.append(combined_feat)
    Y.append(1)

X = np.array(X)
Y = np.array(Y)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
linear_svc = LinearSVC()
linear_svc.fit(X_train_scaled, Y_train)
Y_pred_svc = linear_svc.predict(X_test_scaled)

# Evaluation
OA = accuracy_score(Y_test, Y_pred_svc)
print("Overall Accuracy (OA):", OA)
print("Linear SVM Classification Report:")
print(classification_report(Y_test, Y_pred_svc))