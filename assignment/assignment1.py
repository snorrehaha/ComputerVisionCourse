import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





#Animal recognition Bear / Crocodile
bear_paths = "../Datasets/train/Bear"
crocodile_paths = "../Datasets/train/Crocodile"

bear_paths = [os.path.join(bear_paths, f) for f in os.listdir("../Datasets/train/Bear")]
crocodile_paths = [os.path.join(crocodile_paths, f) for f in os.listdir("../Datasets/train/Crocodile")]

# Load and preprocess image
# Resize, convert to LAB and normalize to [1,0]
def LoadImage(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.astype(np.float32) / 255.0
    return img

# Get colour information: (std and mean) on a and b channels
def GetColorFeatures(img):
    L, a, b = cv2.split(img)
    return np.array([np.mean(a), np.std(a), np.mean(b), np.std(b)])


# Get wavelets features
def GetWaveletFeatures(img):
    L, a, b = cv2.split(img)
    gray = L
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    # Energy features
    features = [np.mean(cA), np.std(cA),
                np.mean(cH), np.std(cH),
                np.mean(cV), np.std(cV),
                np.mean(cD), np.std(cD)]
    return np.array(features)

# Get Sobel features
def GetSobelFeatures(img):
    # Use L channel for intensity/edges
    L, a, b = cv2.split(img)
    gray = L

    # Compute x and y gradients using Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude and orientation
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = np.arctan2(grad_y, grad_x)  # in radians

    # Summarize with simple statistics
    features = [
        np.mean(magnitude),
        np.std(magnitude),
        np.mean(orientation),
        np.std(orientation)
    ]

    return np.array(features)


# Main Program
X = []
Y = []

# Load image
# Get Features and combine them
for path in bear_paths:
    img = LoadImage(path)
    color_features = GetColorFeatures(img)
    wavelet_features = GetWaveletFeatures(img)
    sobel_features = GetSobelFeatures(img)
    combined = np.concatenate([color_features, wavelet_features, sobel_features])
    X.append(combined)
    Y.append(0)

for path in crocodile_paths:
    img = LoadImage(path)
    color_features = GetColorFeatures(img)
    wavelet_features = GetWaveletFeatures(img)
    sobel_features = GetSobelFeatures(img)
    combined = np.concatenate([color_features, wavelet_features, sobel_features])
    X.append(combined)
    Y.append(1)


# Split: 60% train, 20% validation, 20% test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=20)
X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=20)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# Linear SVC, Only on training set
linear_svc = LinearSVC()
linear_svc.fit(X_train_scaled, Y_train)
Y_pred_svc = linear_svc.predict(X_test_scaled)

OA = accuracy_score(Y_test, Y_pred_svc)
print("Baseline Linear SVM OA (train only):", OA)
print(classification_report(Y_test, Y_pred_svc))

# Hyperparameter tuning using validation set
best_score = 0
best_C = None

for C in [0.01, 0.1, 1, 10]:
    svc = LinearSVC(C=C, max_iter=5000)
    svc.fit(X_train_scaled, Y_train)
    score = svc.score(X_validation_scaled, Y_validation)  # validation accuracy
    if score > best_score:
        best_score = score
        best_C = C

print(f"Best C based on validation: {best_C}, validation OA: {best_score:.3f}")

# Retrain on train + validation set with best C
X_trainval_scaled = scaler.fit_transform(np.vstack([X_train, X_validation]))
Y_trainval = np.hstack([Y_train, Y_validation])

best_svc = LinearSVC(C=best_C, max_iter=5000)
best_svc.fit(X_trainval_scaled, Y_trainval)

# Scale test set according to combined train+validation
X_test_scaled_final = scaler.transform(X_test)

# Final test evaluation
Y_pred_test = best_svc.predict(X_test_scaled_final)
OA_final = accuracy_score(Y_test, Y_pred_test)
print("Final Linear SVM OA (train + validation):", OA_final)
print(classification_report(Y_test, Y_pred_test))


# Plotting and visualisation
# Example images
fig, axes = plt.subplots(2, 3, figsize=(10,6))
for i, path in enumerate(bear_paths[:3]):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0,i].imshow(img)
    axes[0,i].set_title('Bear')
    axes[0,i].axis('off')

for i, path in enumerate(crocodile_paths[:3]):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[1,i].imshow(img)
    axes[1,i].set_title('Crocodile')
    axes[1,i].axis('off')

# Feature visualisation: Gradiant features
plt.suptitle("Sample Images from Dataset")
plt.show()

img = cv2.imread(bear_paths[0])
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(img_lab)

grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

plt.figure(figsize=(5,5))
plt.imshow(magnitude, cmap='gray')
plt.title("Sobel Gradient Magnitude (Bear)")
plt.axis('off')
plt.show()

# Wavelet features
coeffs = pywt.dwt2(L, 'haar')
cA, (cH, cV, cD) = coeffs

fig, axes = plt.subplots(1,4, figsize=(15,4))
for ax, feature, title in zip(axes, [cA, cH, cV, cD], ['Approx', 'Horizontal', 'Vertical', 'Diagonal']):
    ax.imshow(feature, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.suptitle("Wavelet Coefficients (Bear Image)")
plt.show()

# Feature statistics
# Example: histogram of gradient magnitude for bears vs crocodiles
grad_magnitudes = []

for path in bear_paths:
    img = cv2.imread(path)
    L, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitudes.append(mag.mean())

plt.hist(grad_magnitudes, bins=10, alpha=0.7, label='Bears')
# Repeat for Crocodiles
grad_magnitudes_croc = []
for path in crocodile_paths:
    img = cv2.imread(path)
    L, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitudes_croc.append(mag.mean())
plt.hist(grad_magnitudes_croc, bins=10, alpha=0.7, label='Crocodiles')

plt.xlabel("Mean Gradient Magnitude")
plt.ylabel("Number of Images")
plt.legend()
plt.title("Distribution of Gradient Magnitudes by Class")
plt.show()

#Confusion matrix
cm = confusion_matrix(Y_test, Y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bear', 'Crocodile'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Feature importance
feature_names = ['a_mean','a_std','b_mean','b_std',
                 'cA_mean','cA_std','cH_mean','cH_std','cV_mean','cV_std','cD_mean','cD_std',
                 'grad_mag_mean','grad_mag_std','grad_ori_mean','grad_ori_std']

weights = np.abs(best_svc.coef_[0])
plt.figure(figsize=(10,4))
plt.bar(feature_names, weights)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Weight Magnitude")
plt.title("LinearSVC Feature Importance")
plt.show()