import numpy as np
import cv2
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import os

# --- Paths ---
bear_paths = "../Datasets/train/Bear"
crocodile_paths = "../Datasets/train/Crocodile"

bear_paths = [os.path.join(bear_paths, f) for f in os.listdir(bear_paths)]
crocodile_paths = [os.path.join(crocodile_paths, f) for f in os.listdir(crocodile_paths)]


# --- Image loader ---
def LoadImage(path, size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.astype(np.float32) / 255.0
    return img


# --- Feature extractors ---
def GetSobelFeatures(img):
    L, a, b = cv2.split(img)
    gray = L
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = np.arctan2(grad_y, grad_x)
    return np.array([np.mean(magnitude), np.std(magnitude),
                     np.mean(orientation), np.std(orientation)])


def GetColorFeatures(img):
    L, a, b = cv2.split(img)
    return np.array([np.mean(a), np.std(a), np.mean(b), np.std(b)])


def GetWaveletFeatures(img):
    L, a, b = cv2.split(img)
    gray = L
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    return np.array([np.mean(cA), np.std(cA),
                     np.mean(cH), np.std(cH),
                     np.mean(cV), np.std(cV),
                     np.mean(cD), np.std(cD)])


# --- Build datasets ---
def BuildDataset(feature_type="combined"):
    X, Y = [], []
    for path in bear_paths:
        img = LoadImage(path)
        if feature_type == "sobel":
            feat = GetSobelFeatures(img)
        elif feature_type == "color":
            feat = GetColorFeatures(img)
        elif feature_type == "wavelet":
            feat = GetWaveletFeatures(img)
        else:  # combined
            feat = np.concatenate([GetSobelFeatures(img),
                                   GetColorFeatures(img),
                                   GetWaveletFeatures(img)])
        X.append(feat)
        Y.append(0)
    for path in crocodile_paths:
        img = LoadImage(path)
        if feature_type == "sobel":
            feat = GetSobelFeatures(img)
        elif feature_type == "color":
            feat = GetColorFeatures(img)
        elif feature_type == "wavelet":
            feat = GetWaveletFeatures(img)
        else:
            feat = np.concatenate([GetSobelFeatures(img),
                                   GetColorFeatures(img),
                                   GetWaveletFeatures(img)])
        X.append(feat)
        Y.append(1)
    return np.array(X), np.array(Y)


# --- Train, validate, test ---
results = {}

for feature_type in ["sobel", "color", "wavelet", "combined"]:
    print(f"\n=== Feature set: {feature_type.upper()} ===")
    X, Y = BuildDataset(feature_type)

    # Train 60%, Validation 20%, Test 20%
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning (C) on validation
    best_score = 0
    best_C = None
    for C in [0.01, 0.1, 1, 10]:
        svc = LinearSVC(C=C, max_iter=5000)
        svc.fit(X_train_scaled, Y_train)
        score = svc.score(X_validation_scaled, Y_validation)
        if score > best_score:
            best_score = score
            best_C = C

    # Retrain on train+validation
    X_trainval_scaled = scaler.fit_transform(np.vstack([X_train, X_validation]))
    Y_trainval = np.hstack([Y_train, Y_validation])
    best_svc = LinearSVC(C=best_C, max_iter=5000)
    best_svc.fit(X_trainval_scaled, Y_trainval)

    # Evaluate on test
    X_test_scaled_final = scaler.transform(X_test)
    Y_pred_test = best_svc.predict(X_test_scaled_final)

    OA = accuracy_score(Y_test, Y_pred_test)
    results[feature_type] = OA

    print(f"Best C: {best_C}, Test OA: {OA:.3f}")
    print(classification_report(Y_test, Y_pred_test))

# --- Summary ---
print("\n=== Overall Accuracy comparison ===")
for feat, oa in results.items():
    print(f"{feat.upper():10s}: OA = {oa:.3f}")