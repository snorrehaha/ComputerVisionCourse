import cv2
import numpy as np
import functions as f

def GetGaborFeatures(img_gray):

    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    kernels = []
    features = []

    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [2, 4]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=sigma,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            kernels.append(kernel)

    responses = []

    for k in kernels:
        filtered = cv2.filter2D(img_gray, cv2.CV_32F, k)
        responses.append(filtered)

        # summarize response
        features.append(np.mean(filtered))
        features.append(np.std(filtered))

    return np.array(features), responses

def ShowGaborResponses(img, responses, title="Gabor Responses"):
    import matplotlib.pyplot as plt

    n = len(responses)

    plt.figure(figsize=(12, 6))
    plt.suptitle(title)

    for i, r in enumerate(responses):
        plt.subplot(2, 4, i + 1)
        plt.imshow(r, cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

img_healthy = f.LoadImage('Potato_healthy/000_healthy.JPG')
img_late = f.LoadImage('Potato_Late_blight/000_late_blight.JPG')
img_early = f.LoadImage('Potato_Early_blight/000_early_blight.JPG')

images = [img_healthy, img_early, img_late]
titles = ["Healthy", "Early Blight", "Late Blight"]

gabor_features = []

for img, title in zip(images, titles):

    # Step 1: segmentation (optional but recommended)
    mask = f.segment_plant_hsv(img)
    segmented = f.apply_mask(img, mask)

    # Step 2: grayscale conversion
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)

    # Step 3: extract features + responses
    feats, responses = GetGaborFeatures(gray)

    # Step 4: visualize responses
    ShowGaborResponses(gray, responses, title=title)

    # Step 5: print features
    print(f"\n{title} Gabor Features:")
    for i, v in enumerate(feats):
        print(f"Feature {i}: {v:.4f}")

    gabor_features.append(feats)