# classification of potato leaf disease
"""Imports"""


import functions as f
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import exposure


"""Pathing and preparing dataset-------------------------------------------------------------------------------------"""
import os

healthy_path = "Potato_healthy/"
early_path   = "Potato_Early_blight/"
late_path    = "Potato_Late_blight/"

healthy_paths = [os.path.join(healthy_path, f) for f in os.listdir(healthy_path)]
early_paths   = [os.path.join(early_path, f)   for f in os.listdir(early_path)]
late_paths    = [os.path.join(late_path, f)    for f in os.listdir(late_path)]

def load_dataset(paths, label):
    X_local = []
    y_local = []

    for path in paths:
        img = f.LoadImage(path)
        #mask = f.segment_plant_hsv(img)

        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)


        feats = f.extract_features(img, mask)



        X_local.append(feats)
        y_local.append(label)

    return X_local, y_local


X, y = [], []

for paths, label in [
    (healthy_paths, 0),
    (early_paths, 1),
    (late_paths, 2)
]:
    X_part, y_part = load_dataset(paths, label)
    X.extend(X_part)
    y.extend(y_part)

X = np.array(X, dtype=np.float32)
y = np.array(y)

"""TRAINING----------------------------------------------------------------------------------------------------------"""

"""Train test val split"""
from sklearn.model_selection import train_test_split
# First split: train vs temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=20)
# Second split: validation vs test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,test_size=0.50, stratify=y_temp,random_state=20)


"""-------------------------Scaler--------------------------------"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


"""----------------------Baseline model---------------------------"""
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

baseline = LinearSVC(max_iter=5000)
baseline.fit(X_train_scaled, y_train)

val_pred = baseline.predict(X_val_scaled)

print("Baseline Validation Performance")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

"""------------------Hyperparamter Tuning with C search------------"""
best_score = 0
best_C = None

for C in [0.01, 0.1, 1, 10]:
    model = LinearSVC(C=C, max_iter=5000)
    model.fit(X_train_scaled, y_train)

    score = model.score(X_val_scaled, y_val)

    if score > best_score:
        best_score = score
        best_C = C

print("Best C:", best_C)
print("Best validation accuracy:", best_score)


"""---------------------Train final model---------------------------"""
import numpy as np

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])

scaler_final = StandardScaler()
X_trainval_scaled = scaler_final.fit_transform(X_trainval)
X_test_scaled_final = scaler_final.transform(X_test)

final_model = LinearSVC(C=best_C, max_iter=5000)
final_model.fit(X_trainval_scaled, y_trainval)

"""---------------------Final evaluation--------------------------"""
test_pred = final_model.predict(X_test_scaled_final)

print("Final Test Results")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

#confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
classes = ["Healthy", "Early", "Late"]
cm = confusion_matrix(y_test, test_pred)
print(cm)

"""Visualization-----------------------------------------------------------------------------------------------------"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

# --- LDA projection ---
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_lda, y)

# --- Meshgrid ---
margin = 1.0
x_min, x_max = X_lda[:, 0].min() - margin, X_lda[:, 0].max() + margin
y_min, y_max = X_lda[:, 1].min() - margin, X_lda[:, 1].max() + margin

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

grid = np.c_[xx.ravel(), yy.ravel()]

# --- Probability-based region shading ---
# proba shape: (N, 3) — one column per class
proba = clf.predict_proba(grid)
Z_pred = np.argmax(proba, axis=1)
Z_conf = np.max(proba, axis=1)   # confidence = how sure the classifier is

Z_pred = Z_pred.reshape(xx.shape)
Z_conf = Z_conf.reshape(xx.shape)

# Class colors (regions and scatter share the same palette)
CLASS_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}  # green, orange, red
CLASS_NAMES  = {0: "Healthy", 1: "Early Blight", 2: "Late Blight"}

# Build an RGBA image for the region background
rgba = np.zeros((*xx.shape, 4))
for label, hex_color in CLASS_COLORS.items():
    rgb = mcolors.to_rgb(hex_color)
    mask = (Z_pred == label)
    rgba[mask, 0] = rgb[0]
    rgba[mask, 1] = rgb[1]
    rgba[mask, 2] = rgb[2]

# Alpha driven by confidence: low-confidence (overlap) zones fade out
rgba[..., 3] = np.clip((Z_conf - 0.33) / 0.67, 0.05, 0.45)

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 7))

ax.imshow(
    rgba,
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    aspect="auto",
    interpolation="bilinear"
)

# Decision boundary lines (where classifier switches prediction)
ax.contour(
    xx, yy, Z_pred,
    levels=[0.5, 1.5],
    colors="black",
    linewidths=1.2,
    linestyles="--"
)

# Covariance ellipses (1σ and 2σ) per class
def plot_cov_ellipse(ax, points, color, n_std=2.0, alpha=0.18):
    mean = points.mean(axis=0)
    cov  = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    for std in [1.0, n_std]:
        w, h = 2 * std * np.sqrt(vals)
        ell = Ellipse(
            xy=mean, width=w, height=h, angle=angle,
            facecolor=color, alpha=alpha if std == n_std else alpha * 0.6,
            edgecolor=color, linewidth=1.5, linestyle="-"
        )
        ax.add_patch(ell)

for label, color in CLASS_COLORS.items():
    pts = X_lda[y == label]
    plot_cov_ellipse(ax, pts, color)

# Scatter points
for label, color in CLASS_COLORS.items():
    idx = (y == label)
    ax.scatter(
        X_lda[idx, 0], X_lda[idx, 1],
        label=CLASS_NAMES[label],
        color=color,
        edgecolor="white",
        linewidth=0.5,
        s=30,
        zorder=5
    )

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title("LDA Projection — Decision Regions & Class Separability", fontsize=13)
ax.set_xlabel("LD1")
ax.set_ylabel("LD2")
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()