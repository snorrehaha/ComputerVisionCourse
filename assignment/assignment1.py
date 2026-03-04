import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------------------------------
#Dataset Paths
bear_paths = "../DatasetForGit/Bear/"
crocodile_paths = "../DatasetForGit/Crocodile/"

bear_paths = [os.path.join(bear_paths, f) for f in os.listdir("../DatasetForGit/Bear/")]
crocodile_paths = [os.path.join(crocodile_paths, f) for f in os.listdir("../DatasetForGit/Crocodile/")]

print(len(bear_paths))
print(len(crocodile_paths))

#Functions
#------------------------------------------------------------------------------------------------------
# Load and preprocess image
# Resize, convert to CIELAB and normalize to [0,1]
def LoadImage(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.astype(np.float32) / 255.0
    return img

# Extract colour statistics: mean and standard deviation of a and b channels
def GetColorFeatures(img):
    L, a, b = cv2.split(img)
    return np.array([np.mean(a), np.std(a), np.mean(b), np.std(b)])

# Extract frequency features: mean and standard deviation of Fourier magnitude spectrum
def GetFourierFeatures(img):
    L, a, b = cv2.split(img)
    gray = L
    # Compute Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    # Magnitude spectrum
    magnitude = np.abs(fshift)
    # Log scaling (important for stability)
    magnitude = np.log(1 + magnitude)
    # Simple statistical features
    features = [
        np.mean(magnitude),
        np.std(magnitude)
    ]
    return np.array(features)

# Extract gradient features: statistics of Sobel magnitude and orientation
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
# ----------------------------------------------------------------------------------------------
X = []
Y = []

# Extract features for each image and build dataset
for path in bear_paths:
    img = LoadImage(path)
    color_features = GetColorFeatures(img)
    fourier_features = GetFourierFeatures(img)
    sobel_features = GetSobelFeatures(img)
    combined = np.concatenate([color_features, fourier_features, sobel_features])
    X.append(combined)
    Y.append(0)

for path in crocodile_paths:
    img = LoadImage(path)
    color_features = GetColorFeatures(img)
    fourier_features = GetFourierFeatures(img)
    sobel_features = GetSobelFeatures(img)
    combined = np.concatenate([color_features, fourier_features, sobel_features])
    X.append(combined)
    Y.append(1)


# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)


# Split: 60% train, 20% validation, 20% test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=20)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=20)


# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------
# Baseline model

linear_svc = LinearSVC(max_iter=5000)
linear_svc.fit(X_train_scaled, Y_train)

Y_pred_val = linear_svc.predict(X_validation_scaled)

print("Baseline Linear SVM (Validation Set)")
print(f"Accuracy: {accuracy_score(Y_validation, Y_pred_val):.3f}")
print(classification_report(Y_validation, Y_pred_val))


# ---------------------------------------
# Hyperparameter tuning
best_score = 0
best_C = None

for C in [0.01, 0.1, 1, 10]:
    svc = LinearSVC(C=C, max_iter=5000)
    svc.fit(X_train_scaled, Y_train)

    score = svc.score(X_validation_scaled, Y_validation)

    if score > best_score:
        best_score = score
        best_C = C

print(f"Best C: {best_C} (Validation Accuracy: {best_score:.3f})")


# ---------------------------------------
# Retrain on train + validation

X_trainval = np.vstack([X_train, X_validation])
Y_trainval = np.hstack([Y_train, Y_validation])

# Refit scaler on combined training and validation data before final evaluation
scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled_final = scaler_final.transform(X_test)

best_svc = LinearSVC(C=best_C, max_iter=5000)
best_svc.fit(X_trainval_scaled, Y_trainval)


# ---------------------------------------
# Final test evaluation
Y_pred_test = best_svc.predict(X_test_scaled_final)

print("Final Test Results (unseen data):")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_test):.3f}")
print(classification_report(Y_test, Y_pred_test))

print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_test))


# Plotting and visualisation
# -----------------------------------------------------------------------------
# Example images
fig, axes = plt.subplots(2, 3, figsize=(10,6))

# Bears
for i, path in enumerate(bear_paths[10:13]):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(img)
    axes[0, i].set_title("Bear")
    axes[0, i].axis("off")

# Crocodiles
for i, path in enumerate(crocodile_paths[:3]):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(img)
    axes[1, i].set_title("Crocodile")
    axes[1, i].axis("off")

plt.suptitle("Sample Dataset Images")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# CIELAB channels
img = LoadImage(bear_paths[10])
L, a, b = cv2.split(img)

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(L, cmap='gray')
plt.title("L (Lightness)")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(a, cmap='gray')
plt.title("a (Green-Red)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(b, cmap='gray')
plt.title("b (Blue-Yellow)")
plt.axis('off')

plt.show()

# ----------------------------------------------------------------------------
# Sobel Gradients
img = LoadImage(bear_paths[10])
L, _, _ = cv2.split(img)

grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(grad_x**2 + grad_y**2)

plt.imshow(magnitude, cmap='gray')
plt.title("Sobel Gradient Magnitude")
plt.axis('off')
plt.show()

# ----------------------------------------------------------------------------
# Fourier magnitude spectrum
img = LoadImage(bear_paths[10])
L, _, _ = cv2.split(img)

f = np.fft.fft2(L)
fshift = np.fft.fftshift(f)
magnitude = np.log(1 + np.abs(fshift))

plt.imshow(magnitude, cmap='gray')
plt.title("Fourier Magnitude Spectrum")
plt.axis('off')
plt.show()

# -------------------------------------------------------------------------
# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred_test)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# ---------------------------------------------------------------------------
# Feature importance
weights = best_svc.coef_[0]

feature_names = [
    "mean(a)", "std(a)", "mean(b)", "std(b)",
    "fourier_mean", "fourier_std",
    "grad_mag_mean", "grad_mag_std",
    "orientation_mean", "orientation_std"
]

plt.figure(figsize=(10,5))
plt.bar(range(len(weights)), weights)

plt.xticks(range(len(weights)), feature_names, rotation=45, ha='right')
plt.title("Feature Importance (Linear SVM Weights)")
plt.xlabel("Features")
plt.ylabel("Weight")

plt.tight_layout()
plt.show()
